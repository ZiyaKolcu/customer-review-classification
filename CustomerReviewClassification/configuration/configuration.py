from CustomerReviewClassification.entity.config_entity import ETL_Config
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.constants import *
from CustomerReviewClassification.utils.common import read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_file_path=CONFIG_FILE_PATH,
    ):
        self.config = read_yaml(config_file_path)

    def get_etl_config(self) -> ETL_Config:
        config = self.config.etl
        etl_config = ETL_Config(raw_data_dir=config.raw_data_dir, table_name=config.table_name)

        return etl_config
