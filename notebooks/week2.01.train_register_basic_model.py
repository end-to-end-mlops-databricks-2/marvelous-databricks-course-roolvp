# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

from alzheimers_prediction.config import ProjectConfig, Tags
from alzheimers_prediction.models.basic_model import BasicModel

# COMMAND ----------
# Default profile
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()

tags = Tags(**{"git_sha":"ccdd47f458bf3c29586e7614260d14e764efd35d", "branch":"week2-MLFlow-Feature-Eng"})

# COMMAND ----------

BasicModel(config=config, tags=tags, spark=spark).load_data()


# COMMAND ----------
