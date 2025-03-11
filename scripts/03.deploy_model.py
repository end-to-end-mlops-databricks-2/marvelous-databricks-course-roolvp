from argparse import ArgumentParser

from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

from alzheimers_prediction.config import ProjectConfig
from alzheimers_prediction.serving.model_serving import ModelServing
from typing import List, Dict
from loguru import logger

import mlflow

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


spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"



# Load project configuration
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
catalog_name = config.catalog_name
schema_name = config.schema_name

logger.info("Deploying the model serving endpoint...")
# Initialize the model serving manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.alzheimers_prediction_custom_model",
    endopoint_name="alzheimers_prediction_custom_model"
)
model_serving.deploy_or_update_serving_endpoint(version="latest")
logger.info("Endpoint deployment started...")

