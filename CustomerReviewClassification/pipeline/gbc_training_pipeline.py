from CustomerReviewClassification.configuration.configuration import (
    ConfigurationManager,
)
from CustomerReviewClassification.components.data_ingestion import DataIngestion
from CustomerReviewClassification.components.data_validation import DataValidation
from CustomerReviewClassification.components.data_transformation import (
    DataTransformation,
)
from CustomerReviewClassification.components.model_evaluation import ModelEvaluation
from CustomerReviewClassification.components.gbc_model_training import GBC_ModelTraining
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.logging.logger import logging
from pathlib import Path
import json
import sys


class GBC_Training_Pipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def initiate_data_ingestion(self):
        data_ingestion_config = self.config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        df = data_ingestion.export_database_table_as_dataframe()
        data_ingestion.export_data_into_file(dataframe=df)

    def initiate_data_validation(self):
        data_validation_config = self.config_manager.get_data_validation_config()
        data_validation = DataValidation(data_validation_config=data_validation_config)
        data_validation.validate_all_columns()

    def initiate_data_transform(self):
        try:
            with open(Path("artifacts/data_validation/status.json"), "r") as file:
                status = json.load(file)

            validation_status = status.get("Validation status")
            if validation_status == True:
                data_transform_config = (
                    self.config_manager.get_data_transformation_config()
                )
                data_transform = DataTransformation(
                    data_transform_config=data_transform_config
                )
                data_transform.transform()

            else:
                raise CustomException("Data schema is not valid", sys)

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_gbc_model_training(self):
        gbc_model_training_config = self.config_manager.get_gbc_model_training_config()
        gbc_model_training = GBC_ModelTraining(
            gbc_model_training_config=gbc_model_training_config
        )
        gbc_model_training.train()

    def initiate_model_evaluation(self):
        model_evaluation_config = self.config_manager.get_model_evaluation_config(
            selected_model="GBC"
        )
        model_evaluation = ModelEvaluation(
            model_evaluation_config=model_evaluation_config
        )
        model_evaluation.log_into_mlflow()
