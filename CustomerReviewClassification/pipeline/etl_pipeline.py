from CustomerReviewClassification.configuration.configuration import (
    ConfigurationManager,
)
from CustomerReviewClassification.components.etl import ETL


class ETL_Pipeline:
    def __init__(self):
        pass

    def initiate_etl(self):
        config_manager = ConfigurationManager()
        etl_config = config_manager.get_etl_config()
        etl = ETL(etl_config=etl_config)
        etl.load()
