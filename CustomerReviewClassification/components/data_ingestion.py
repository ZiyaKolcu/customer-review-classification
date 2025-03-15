import os
import sys
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.entity.config_entity import DataIngestionConfig
from dotenv import load_dotenv

load_dotenv()


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

        try:
            self.conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST"),
                port=os.getenv("POSTGRES_PORT"),
                database=os.getenv("POSTGRES_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
            )
            self.cur = self.conn.cursor()
            logging.info("Connected to the db")
        except Exception as e:
            raise CustomException(e, sys)

    def export_database_table_as_dataframe(self):
        """
        Read data from PosgreSQL
        """
        try:
            query = f"SELECT * FROM {self.data_ingestion_config.table_name};"
            df = pd.read_sql(query, self.conn)

            if "id" in df.columns.to_list():
                df = df.drop(columns=["id"], axis=1)
            self.conn.close()

            logging.info("Data exported successfully!")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_file(
        self, dataframe: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ):
        try:
            data_dir = self.data_ingestion_config.root_dir

            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)

            train_df, test_df = train_test_split(
                dataframe, test_size=test_size, random_state=random_state
            )

            train_path = os.path.join(data_dir, "train.csv")
            test_path = os.path.join(data_dir, "test.csv")

            train_df.to_csv(train_path, index=False, header=True)
            test_df.to_csv(test_path, index=False, header=True)

            logging.info(f"Train data saved successfully at {train_path}")
            logging.info(f"Test data saved successfully at {test_path}")

            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)
