# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

from alzheimers_prediction.config import ProjectConfig, Tags
from alzheimers_prediction.models.basic_model import BasicModel

# COMMAND ----------
# Default profile (locally use the databricks cli to set it)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

mlflow.login()
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()

tags = Tags(**{"git_sha":"ccdd47f458bf3c29586e7614260d14e764efd35d", "branch":"week2-MLFlow-Feature-Eng"})

# COMMAND ----------

basic_model = BasicModel(config=config, tags=tags, spark=spark)
basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------

basic_model.train()
# COMMAND ----------

basic_model.log_model()
# COMMAND ----------

run_id = mlflow.search_runs(
    experiment_names=[config.experiment_name_basic], filter_string="tags.branch = 'week2-MLFlow-Feature-Eng'"
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/logisticreg-pipeline-model")

# COMMAND ----------

basic_model.retrieve_current_run_dataset()

# COMMAND ----------
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
basic_model.register_model()

# COMMAND ----------

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.toPandas()[config.num_features + config.cat_features]

predictions_df = basic_model.load_latest_model_and_predict(X_test)


# COMMAND ----------
predictions_df
# COMMAND ----------
