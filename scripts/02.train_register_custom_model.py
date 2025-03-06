import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from alzheimers_prediction.config import ProjectConfig, Tags
from alzheimers_prediction.models.custom_model import CustomModel


mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "ccdd47f458bf3c29586e7614260d14e764efd35d", "branch": "week2-MLFlow-Feature-Eng"})



# [Q] What is code_paths? The location of the wheel files of the custom packages that are needed to run the model.
custom_model = CustomModel(
    config=config, tags=tags, spark=spark, code_paths=["../dist/alzheimers_prediction-0.0.3-py3-none-any.whl"]
)


custom_model.load_data()
custom_model.prepare_features()


custom_model.train()
custom_model.log_model()


run_id = mlflow.search_runs(
    experiment_names=[config.experiment_name_custom], filter_string="tags.branch = 'week2-MLFlow-Feature-Eng'"
).run_id[0]


custom_model.retrieve_current_run_dataset()


custom_model.retrieve_current_run_metadata()


# Register the model in the Databricks Unity Catalog
custom_model.register_model()



test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)

X_test = test_set.toPandas()[config.num_features + config.cat_features]

logger.info(f"X_test sample: {X_test.sample(1)}")

custom_prediction_response = custom_model.load_latest_model_and_predict(X_test.sample(1))

logger.info(f"Custom prediction response: {custom_prediction_response}")