from typing import List

import mlflow
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from alzheimers_prediction.config import ProjectConfig, Tags


class AlzheimersPredictionModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        """
        LESSON: The context needs to be passed, even if not used here.
            The context argument will be passed by DataBricks when the model is deployed.
            If the argument is not present, the wrapper will error.
            The
        """
        # Get the class predictions (0 or 1)
        predictions: int = self.model.predict(model_input)

        # Get the probabilities for each class (0 and 1)
        probabilities = self.model.predict_proba(model_input)

        # Get the probabilities for class 1 (Alzheimer's diagnosis)
        positive_diagnosis_probability = probabilities[0][1]

        return {"positive_diagnosis_probability": positive_diagnosis_probability, "prediction": predictions[0]}


class CustomModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: List[str]):
        """
        Code paths: List of paths to the wheel files of the project
        """

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
        self.experiment_name = self.config.experiment_name_custom
        self.model_name = f"{self.catalog_name}.{self.schema_name}.alzheimers_prediction_custom_model"

        # LESSON: Code paths are needed. Why?
        self.code_paths = code_paths

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
        # LESSON: Pyspark is needed for the serving endpoint because it is imported in this package
        additional_pip_deps = ["pyspark==3.5.0"]
        # LESSON: code_paths is a list of paths (locations) to the custom wheel files of the custom packages used by the model
        # LESSON: This way Databricks can install the custom packages when the model is deployed
        # How do we know the model has been registered correctly? Check it in Unity Catalog
        # Adding "code/" to the path is necessary because the custom packages are stored in the code directory. _mlflow_conda_env expexts this format
        for package in self.code_paths:
            whl_name = package.split("/")[-1]
            additional_pip_deps.append(f"code/{whl_name}")

        logger.info(f"📦 Custom packages to be installed: {additional_pip_deps}")
        # LESSON: Config to manage the custom packages. DBX uses condas to install the custom packages
        conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

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
            signature = infer_signature(
                model_input=self.X_test, model_output={"positive_diagnosis_probability": 0.2, "prediction": 0}
            )
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version,
            )

            mlflow.log_input(dataset=dataset, context="training")

            # LESSON: The model is wrapped in a class that has a predict method
            # LESSON: the contex argument is still None
            # LESSON: code_paths is a list of paths (locations) to the custom wheel files of the custom packages used by the model

            # LESSON: This is a pyfunc model. Not a sklearn model.
            # LESSON: This uploads the wheel file of the custom package to Unity Catalog
            # This way the pacakge is available. Doing it as a private package does not work.
            # Databricks needs the "code/" folder. That' the only way for serving

            mlflow.pyfunc.log_model(
                python_model=AlzheimersPredictionModelWrapper(self.pipeline),
                artifact_path="pyfunc-logisticreg-pipeline-model",
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature,
            )

    def register_model(self):
        """
        Register the model in Databricks Unity Catalog
        """
        logger.info("📦 Registering the model in the Databricks Unity Catalog")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pyfunc-logisticreg-pipeline-model", name=self.model_name, tags=self.tags
        )

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(name=self.model_name, alias="latest-model", version=latest_version)

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

        logger.info("🔮 Loading the latest model from MLflow alias 'latest-model' ")

        model_uri = f"models:/{self.model_name}@latest-model"

        model: AlzheimersPredictionModelWrapper = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Model loaded successfully")

        # LESSON: The model is wrapped in a class that has a predict method
        # LESSON: the contex argument is None
        predictions = model.predict(input_data)

        return predictions
