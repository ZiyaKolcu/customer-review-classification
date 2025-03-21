from CustomerReviewClassification.configuration.configuration import (
    ConfigurationManager,
)
from CustomerReviewClassification.components.data_ingestion import DataIngestion
from CustomerReviewClassification.logging.logger import logging


class LR_Training_Pipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def initiate_data_ingestion(self):
        data_ingestion_config = self.config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        df = data_ingestion.export_database_table_as_dataframe()
        data_ingestion.export_data_into_file(dataframe=df)
