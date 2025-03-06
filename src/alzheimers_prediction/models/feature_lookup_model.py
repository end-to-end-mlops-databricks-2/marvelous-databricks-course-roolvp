import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient  # Why is this here?
from loguru import logger
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
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


class FeatureLookupModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):  # No code_paths here
        """
        Initialize the FeatureLookupModel class
        """

        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient(model_registry_uri="databricks-uc")

        # Extract model parameters from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.model_name = f"{self.catalog_name}.{self.schema_name}.alzheimers_prediction_feature_model"

        # self.model_name = f"{self.catalog_name}.{self.schema_name}.alzheimers_prediction_feature_model"

        # LESSON: Feature Lookup Table
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.alzheimers_features"

        # LESSON: Feature Custom Function
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_healthy_lifestyle_score"

        # MLflow config
        self.tags = tags.model_dump()
        self.experiment_name = self.config.experiment_name_custom

    def create_feature_table(self):
        # [Q] What are recommendations and good practices for managing Databricks tables in code? Adding new columns and altering the schema
        """
        This is to simulate the scenario of online feature lookup
        """
        # with emoji of mechanic log
        logger.info(f"🔧 Creating the feature lookup table {self.feature_table_name}")
        self.spark.sql(f"""
            CREATE OR REPLACE TABLE {self.feature_table_name} (
                Id STRING NOT NULL,
                smoking_status  STRING,
                alcohol_consumption STRING,
                diabetes    STRING
                );
        """)
        logger.info("✅ Feature lookup table created successfully")

        # LESSON: PK is needed in feature lookup tables. It is also important to have it as string.
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT patient_pk PRIMARY KEY(Id);")
        # LESSON: Enable change data feed allow for keeping track of data versioning. We'll be using it for copy the change in the tables.
        # LESSON: We'll be only recording the change in the table. For example, for tracking change of statuses
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} Select Id, smoking_status, alcohol_consumption, diabetes from {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} Select Id, smoking_status, alcohol_consumption, diabetes from {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("✅ Data inserted into the feature lookup table")

    def define_feature_function(self):
        """
        Feature functions: healthy_lifestyle_score
        """
        logger.info(f"🔧 Defining the feature function {self.function_name}")
        self.spark.sql(f"""
            CREATE OR REPLACE FUNCTION {self.function_name} (physical_activity_level STRING)
            RETURNS INT
            LANGUAGE PYTHON AS
            $$
            if physical_activity_level == 'Low':
                return 0
            elif physical_activity_level == 'Medium':
                return 5
            elif physical_activity_level == 'High':
                return 10
            else:
                return 5
            $$;
                """)
        logger.info("✅ Feature function defined successfully")

    def _recreate_feature_function(self, physical_activity_level: str):
        """
        Recreate the feature function in pure Python. This is to build the training and test sets. Just for the sake of the example
        """
        if physical_activity_level == "Low":
            return 0
        elif physical_activity_level == "Medium":
            return 5
        elif physical_activity_level == "High":
            return 10
        else:
            return 5

    def load_data(self):
        # Dropping columns to simulate that they need to be looked when online
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            "smoking_status", "alcohol_consumption", "diabetes"
        )

        # This is Pandas, the previous one was Spark
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        self.train_set = self.train_set.withColumn("Id", self.train_set["Id"].cast("string"))

        logger.info("🏗️ Data loaded successfully")

    def feature_engineering(self):
        """
        Combine table data + feature lookup + feature function
        """
        # LESSON: Using the FE client

        self.training_set = self.fe.create_training_set(
            df=self.train_set,  # Represents the features the user is going to provide
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["smoking_status", "alcohol_consumption", "diabetes"],
                    lookup_key="Id",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    input_bindings={"physical_activity_level": "physical_activity_level"},
                    output_name="healthy_lifestyle_score",
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        self.test_set["healthy_lifestyle_score"] = self.test_set["physical_activity_level"].apply(
            self._recreate_feature_function
        )

        self.X_train = self.training_df[self.num_features + self.cat_features + ["healthy_lifestyle_score"]]
        self.y_train = self.training_df[self.target]

        self.X_test = self.test_set[self.num_features + self.cat_features + ["healthy_lifestyle_score"]]
        self.y_test = self.test_set[self.target]

        logger.info("🔧 Feature engineering completed successfully")

    def train(self):
        """
        Define pipeline
        Set experiment
        Do a Run
        LESSON: Log the model to the run. It is not the same model registry
        """

        logger.info("⌛ Defining feature pre-processing pipeline")
        self.preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("classifier", LogisticRegression(**self.parameters))]
        )
        logger.info("✅ Feature pre-processing pipeline defined successfully")

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            self.pipeline.fit(self.X_train, self.y_train)
            y_pred = self.pipeline.predict(self.X_test)

            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            logger.info("🎯 Model evaluation metrics:")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1: {f1}")

            mlflow.log_params("model_type", "LogisticRegression with OneHotEncoding and Healthy Lifestyle Score")
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            signature = infer_signature(self.X_train, y_pred)

            # LESSON: Log FE model
            self.fe.log_model(
                model=self.pipeline,
                flavor=mlflow.sklearn,
                artifact_path="logistic-regression-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self):
        """
        Register the model with versionning and tags for dynamic selection
        """
        registered_model = mlflow.register_model(
            model_uri="runs:/{self.run_id}/logistic-regression-model-fe", name=self.model_name, tags=self.tags
        )

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias_name="latest-model",
            version=latest_version,
        )

    def load_latest_model_and_predict(self, X):
        """
        Load the latest model and predict
        """
        model_uri = f"models:/{self.model_name}@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
