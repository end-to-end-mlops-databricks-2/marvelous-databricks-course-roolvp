import mlflow
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

from alzheimers_prediction.config import ProjectConfig, Tags
from alzheimers_prediction.models.custom_model import CustomModel
from argparse import ArgumentParser


mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

parser = ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)


# code_paths: The location of the wheel files (where this script is run) of the custom packages that are needed to run the model.
logger.info(f"Loading custom model with tags: {tags_dict}")

# TODO change this to be dynamic or even remove it and use the example of a wrapped model from AB testing
custom_model = CustomModel(
    config=config, tags=tags, spark=spark, code_paths=[f"{root_path}/dist/alzheimers_prediction-0.0.4-py3-none-any.whl"] 
)

logger.info("Loading data and preparing features...")
custom_model.load_data()
custom_model.prepare_features()

logger.info("Training the model...")
custom_model.train()
custom_model.log_model()


# TODO Bring a test set an compare the model with version "latest-model" (the one that is deployed) versus the one that is trained here and then choose the best one.

# Register the model in the Databricks Unity Catalog
latest_version = custom_model.register_model()

logger.info("Model registered successfully:", latest_version)
# LESSON: Used to pass variables from one job to another
dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
dbutils.jobs.taskValues.set(key="model_updated", value=1) # 1 means that the model was updated. 0 means that the model was not updated.
