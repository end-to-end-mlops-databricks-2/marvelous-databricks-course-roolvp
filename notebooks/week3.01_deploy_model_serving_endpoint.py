# Databricks notebooks source

# COMMAND ----------
import os
import requests
import time

from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

from alzheimers_prediction.config import ProjectConfig
from alzheimers_prediction.serving.model_serving import ModelServing
from typing import List, Dict

import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)


# COMMAND ----------
# Get DBX environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")


# COMMAND ----------
# Load project configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Initialize the model serving manager 

model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.alzheimers_prediction_custom_model",
    endopoint_name="alzheimers_prediction_custom_model"
)

# COMMAND ----------
# Deploy the serving endpoint
model_serving.deploy_or_update_serving_endpoint(version="latest")



# COMMAND ----------
# Testing the endpoint: Sample Request Body
required_columns = config.num_features + config.cat_features

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()


sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]



def call_endpoint(record):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/alzheimers_prediction_custom_model/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},

        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# "load test"

for i in range(len(dataframe_records)):
    response, text = call_endpoint(dataframe_records[i])
    print(dataframe_records[i])
    print(f"Response Status: {text}")
    time.sleep(0.2)
# COMMAND ----------
