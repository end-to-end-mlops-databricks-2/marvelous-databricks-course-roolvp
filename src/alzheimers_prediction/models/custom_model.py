import mlflow
import pandas as pd

from mlflow import MlflowClient
from mlflow.models.signature import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from pyspark.sql import SparkSession

from alzheimers_prediction.config import ProjectConfig, Tags
from loguru import logger


class AlzheimersPredictionModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        """
        LESSON: The context needs to be passed, even if not used here. This will be passed by DataBricks when the model is deployed.
        """
        return self.model.predict(model_input)



class CustomModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        self.config = config
        self.spark = spark

        # Extract model parameters from the config
        self.tags = tags.model_dump()
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.alzheimers_prediction_basic_model"

    def load_data(self) -> None:
        """
        Load date from the train set and test set in the catalog.

        Split the data into features and target
        """
        # Load data from the catalog
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set_pandas = self.train_set_spark.toPandas()
        
        self.test_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_set_pandas = self.test_set_spark.toPandas()
        self.data_version = "0" 

        self.X_train = self.train_set_pandas[self.num_features + self.cat_features]
        self.y_train = self.train_set_pandas[self.target]

        self.X_test = self.test_set_pandas[self.num_features + self.cat_features]
        self.y_test = self.test_set_pandas[self.target]
        logger.info("🏗️ Data loaded successfully")

    def prepare_features(self) -> None:
        """
        Encode categorical features and ignores unseen categories

        preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)

        transformed_df_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=preprocessor.get_feature_names_out())
        transformed_df_test = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out())
        """
        logger.info("⌛ Defining feature pre-processing pipeline")
        self.preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("classifier", LogisticRegression(**self.parameters))]
        )
        logger.info("✅ Feature pre-processing pipeline defined successfully")

    def train(self) -> None:
        """
        Train the model
        """
        logger.info("🏋️ Training the model")
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("✅ Model trained successfully")

    def log_model(self):
        """
        Evaluate the model and log it to MLflow
        """
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            y_pred = self.pipeline.predict(self.X_test)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            logger.info("🎯 Model evaluation metrics:")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1: {f1}")

            mlflow.log_param("model_type", "LogisticRegression with OneHotEncoding")
            mlflow.log_params(self.parameters)
            mlflow.log_metrics({"precision": precision, "recall": recall, "f1": f1})

            # Log the model
            signature = infer_signature(model_input=self.X_test, model_output=y_pred)
            dataset = mlflow.data.from_spark(self.train_set_spark,
                                             table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                                             version=self.data_version)
        
            mlflow.log_input(dataset=dataset, context="training")
            mlflow.sklearn.log_model(self.pipeline, artifact_path="logisticreg-pipeline-model", signature=signature)

    
    def register_model(self):
        """
        Register the model in Databricks Unity Catalog
        """
        logger.info("📦 Registering the model in the Databricks Unity Catalog")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/logisticreg-pipeline-model", 
            name=self.model_name,
            tags=self.tags
            )
        
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version
        )
        

    def retrieve_current_run_dataset(self):
        """
        Retrieve the dataset used in the current run
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        # What  is a dataset? Is it a reference or the actual data?
        logger.info("📊 Dataset source loaded")
        return dataset_source.load()
    
    def retrieve_current_run_metadata(self):

        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("📊 Dataset metadata loaded")
        return metrics, params
    
    def load_latest_model_and_predict(self, input_data: pd.DataFrame):
        """
        Load the latest model from MLflow (alias=latest-model) and make predictions.
        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """

        logger.info("🔮 Loading the latest model from MLflow alias 'production'")

        model_uri = f"models:/{self.model_name}@latest-model"

        model = mlflow.sklearn.load_model(model_uri)
        logger.info("✅ Model loaded successfully")
        
        predictions = model.predict(input_data)

        return predictions
        
    
