import logging

import yaml
from pyspark.sql import SparkSession

from alzheimers_prediction.config import ProjectConfig
from alzheimers_prediction.data_processor import DataProcessor
from argparse import ArgumentParser


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default="..",
    type=str,
    required=False, 
)

parser.add_argument(
    "--env",
    action="store",
    default="dev",
    type=str,
    required=False,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/project_config.yml"

env = args.env

logger.info("Loading configuration from %s", config_path)
config = ProjectConfig.from_yaml(config_path=config_path, env=env)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

logging.basicConfig(level=logging.INFO)


# Load the data
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/alzheimers_prediction_dataset.csv",
    header=True,
    inferSchema=True,
).toPandas()


# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
train_set, test_set = data_processor.split_data()
logger.info("Train set shape %s", train_set.shape)
logger.info("Test set shape %s", test_set.shape)


# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(train_set=train_set, test_set=test_set)
logger.info("Data saved to catalog")


# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()