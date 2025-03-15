from CustomerReviewClassification.entity.config_entity import (
    ETL_Config,
    DataIngestionConfig,
)
from CustomerReviewClassification.constants import *
from CustomerReviewClassification.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(
        self,
        config_file_path=CONFIG_FILE_PATH,
    ):
        self.config = read_yaml(config_file_path)

    def get_etl_config(self) -> ETL_Config:
        config = self.config.etl
        etl_config = ETL_Config(
            raw_data_dir=config.raw_data_dir, table_name=config.table_name
        )

        return etl_config

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            ingested_dir=config.ingested_dir,
            table_name=config.table_name,
        )
        return data_ingestion_config
