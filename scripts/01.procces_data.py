import logging

import yaml
from pyspark.sql import SparkSession

from alzheimers_prediction.config import ProjectConfig
from alzheimers_prediction.data_processor import DataProcessor



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


config = ProjectConfig.from_yaml(config_path="../project_config.yml")

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


