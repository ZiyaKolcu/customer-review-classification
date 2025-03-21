import os
import sys
import json
import pandas as pd
from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig):
        self.config = data_validation_config

    def validate_all_columns(self) -> bool:
        try:
            logging.info("validate all columns started")
            validation_status = None
            train_data = pd.read_csv(self.config.train_dir)
            test_data = pd.read_csv(self.config.test_dir)

            data = pd.concat([train_data, test_data])

            all_cols = list(data.columns)
            all_schema = list(self.config.all_schema.keys())

            validation_status = all([col in all_schema for col in all_cols])

            status_data = {"Validation status": validation_status}
            logging.info(f"Validation Status: {status_data}")

            with open(self.config.status_dir, "w") as f:
                json.dump(status_data, f, indent=4)
                logging.info(f"Validation status file saved at {self.config.root_dir}")

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)
