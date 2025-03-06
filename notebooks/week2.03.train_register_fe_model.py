# Databricks notebook source

import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from alzheimers_prediction.config import ProjectConfig, Tags
from alzheimers_prediction.models.feature_lookup_model import FeatureLookupModel

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha":"ccdd47f458bf3c29586e7614260d14e764efd35d", "branch":"week2-MLFlow-Feature-Eng"})

# config = ProjectConfig.from_yaml(config_path="Volumes/mlops_dev/raulvenp/data/project_config.yml")


# COMMAND ----------
# Initialize model
fe_model = FeatureLookupModel(spark=spark, config=config, tags=tags)

# COMMAND ----------
# Create feature lookup table
fe_model.create_feature_table()

# COMMAND ----------
# Define feature function
fe_model.define_feature_function()


# COMMAND ----------
# Load data
fe_model.load_data()

# COMMAND ----------
# Do some feature engineering magic
fe_model.feature_engineering()

# COMMAND ----------
# Train and log the run
fe_model.train()


# COMMAND ----------
# Register the model in the registry
fe_model.register_model()


# COMMAND ----------
# Getting the latest version of the model from the registry and predict 
spark = SparkSession.builder.getOrCreate()

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)
                       
# artificially drop the columns to simulate the real scenario

X_test = test_set.frop("smoking_status", "alcohol_consumption", "diabetes", config.target)

fe_model = FeatureLookupModel(spark=spark, config=config, tags=tags)

predictions = fe_model.load_latest_model_and_predict(X_test)

logger.info(predictions)