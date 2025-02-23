# Databricks notebook source
# MAGIC %md
# MAGIC # Alzheimers Prediction Data Pre-Processing. The Ugly Way
# MAGIC
# MAGIC
# MAGIC - Pre-Process raw data to a SQL compliant format. No feature engineering here
# MAGIC - Split into train and test tables
# MAGIC - Load to tables in Databricks

# COMMAND ----------

import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# COMMAND ----------

# Load configuration
with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

catalog_name = config["catalog_name"]
schema_name = config["schema_name"]
target = config["target"]
print(catalog_name, schema_name, target)

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Only works in a Databricks environment if the data is there
# to put data there, create volume and run databricks fs cp <path> dbfs:/Volumes/mlops_dev/<schema_name>/<volume_name>/
# filepath = f"/Volumes/{catalog_name}/{schema_name}/data/data.csv"
# Load the data
# df = pd.read_csv(filepath)

# Works both locally and in a Databricks environment
filepath = "../data/alzheimers_prediction_dataset.csv"
# Load the data
df = pd.read_csv(filepath)

# Works both locally and in a Databricks environment
# df = spark.read.csv(f"/Volumes/{catalog_name}/{schema_name}/data/data.csv", header=True, inferSchema=True).toPandas()


# COMMAND ----------

df.dtypes

# COMMAND ----------

# Pre-process the data

# Make columns names SQL compliant
df.columns = [column.replace(" ","_").lower().replace("’","") for column in df.columns]
df.columns = [re.sub(r'[^a-zA-Z0-9]', '_', column) for column in df.columns]

# Cast features to the correct type (Types are different locally than in databrics, locally they are 32, in Databricks they are 64)

num_features = df.select_dtypes(include=['int32','int64','float64','float32']).columns.tolist() 
cat_features = df.select_dtypes(include=['object']).columns.tolist()
cat_features.remove(target)
for cat_col in cat_features:
    df[cat_col] = df[cat_col].astype("category")

# Make target 1 or 0
df[target] = df[target].apply(lambda x: 1 if x == 'Yes' else 0)

# Add df index as id column
df['id'] = df.index.astype(str)

# COMMAND ----------



# COMMAND ----------

df.columns

# COMMAND ----------

relevant_columns = cat_features + num_features + [target] + ['id']
df = df[relevant_columns]
train_set, test_set = train_test_split(df, test_size=0.2)

# COMMAND ----------


train_set.sample(10)

# COMMAND ----------



# COMMAND ----------

# Load to database with update timestamp

# Transform Pandas DF to Spark. If it is not already a Spark DF
if not isinstance(train_set, pd.DataFrame):
    train_set = pd.DataFrame(train_set)
if not isinstance(test_set, pd.DataFrame):
    test_set = pd.DataFrame(test_set)

train_set = spark.createDataFrame(train_set)
train_set_with_timestamp = train_set.withColumn("update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))

test_set = spark.createDataFrame(test_set)
test_set_with_timestamp =  test_set.withColumn("update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))


# COMMAND ----------



# COMMAND ----------

# Write to database
train_set_with_timestamp.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.train_set")
test_set_with_timestamp.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.test_set")
