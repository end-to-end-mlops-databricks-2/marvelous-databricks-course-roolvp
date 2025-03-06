import re

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from alzheimers_prediction.config import ProjectConfig


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession):
        self.df = pandas_df
        self.config = config
        self.spark = spark

    def preprocess(self):
        # Make columns names SQL compliant
        self.df.columns = [column.replace(" ", "_").lower().replace("’", "") for column in self.df.columns]
        self.df.columns = [re.sub(r"[^a-zA-Z0-9]", "_", column) for column in self.df.columns]

        # Cast features to the correct type (Types are different locally than in databrics, locally they are 32, in Databricks they are 64)

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            # Cast to int64 or float64 as necessary
            if pd.api.types.is_integer_dtype(self.df[col]):
                self.df[col] = self.df[col].astype("int64")
            elif pd.api.types.is_float_dtype(self.df[col]):
                self.df[col] = self.df[col].astype("float64")

        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        # Make target 1 or 0
        target = self.config.target
        self.df[target] = self.df[target].apply(lambda x: 1 if x == "Yes" else 0)

        # Add self.df index as id column
        self.df["id"] = self.df.index.astype(str)

        # Column selection
        relevant_columns = cat_features + num_features + [target] + ["id"]
        self.df = self.df[relevant_columns]

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self):
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
