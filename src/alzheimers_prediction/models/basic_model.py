import mlflow
import pandas as pd

from mlflow import MlflowClient
from pyspark.sql import SparkSession

from alzheimers_prediction.config import ProjectConfig, Tags
from loguru import logger

class BasicModel:
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

        self.X_train = self.train_set_pandas[self.num_features + self.cat_features]
        self.y_train = self.train_set_pandas[self.target]

        self.X_test = self.test_set_pandas[self.num_features + self.cat_features]
        self.y_test = self.test_set_pandas[self.target]
        logger.info("🏗️ Data loaded successfully")
