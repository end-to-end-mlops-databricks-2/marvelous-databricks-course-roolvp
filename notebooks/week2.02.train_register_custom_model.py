# Databricks notebook source
import mlflow
from pyspark.sql import SparkSession

from alzheimers_prediction.config import ProjectConfig, Tags
from alzheimers_prediction.models.custom_model import CustomModel

# COMMAND ----------
# MAGIC %md
# MAGIC # Train and Register Custom Model
# MAGIC
# MAGIC Basic MLflow models provide raw prediction outputs from the models. For example, a Boolean `0` or `1`, or an array of numbers. However, when serving those models, we almost always want to provide additional information, such as the probability of the prediction.
# MAGIC
# MAGIC For example:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "positive_diagnosis_probability": 0.8,
# MAGIC   "prediction": 1
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC In this notebook, we will train a custom model that will provide this output.
# MAGIC ## Gotcha s and Tips
# MAGIC


# COMMAND ----------
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "ccdd47f458bf3c29586e7614260d14e764efd35d", "branch": "week2-MLFlow-Feature-Eng"})

# COMMAND ----------

# [Q] What is code_paths? The location of the wheel files of the custom packages that are needed to run the model.
custom_model = CustomModel(
    config=config, tags=tags, spark=spark, code_paths=["../dist/alzheimers_prediction-0.0.3-py3-none-any.whl"]
)

# COMMAND ----------
custom_model.load_data()
custom_model.prepare_features()

# COMMAND ----------
custom_model.train()
custom_model.log_model()

# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=[config.experiment_name_custom], filter_string="tags.branch = 'week2-MLFlow-Feature-Eng'"
).run_id[0]

# COMMAND ----------
custom_model.retrieve_current_run_dataset()

# COMMAND ----------
custom_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register the model in the Databricks Unity Catalog
custom_model.register_model()

# COMMAND ----------

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)

X_test = test_set.toPandas()[config.num_features + config.cat_features]
X_test.sample(1)

custom_prediction_response = custom_model.load_latest_model_and_predict(X_test)

# COMMAND ----------
print(custom_prediction_response)
# COMMAND ----------
