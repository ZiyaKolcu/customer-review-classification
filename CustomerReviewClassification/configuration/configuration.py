import os
from CustomerReviewClassification.entity.config_entity import (
    ETL_Config,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
)
from CustomerReviewClassification.constants import *
from CustomerReviewClassification.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(
        self,
        config_file_path=CONFIG_FILE_PATH,
        schema_file_path=SCHEMA_FILE_PATH,
        params_file_path=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_file_path)
        self.schema = read_yaml(schema_file_path)
        self.params = read_yaml(params_file_path)

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

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            train_dir=config.train_dir,
            test_dir=config.test_dir,
            status_dir=config.status_dir,
            all_schema=schema.COLUMNS,
            oov_threshold=schema.OOV.threshold,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            train_dir=config.train_dir,
            test_dir=config.test_dir,
            preprocessor_dir=config.preprocessor_dir,
        )

        return data_transformation_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            X_train_dir=config.X_train_dir,
            y_train_dir=config.y_train_dir,
            params=self.params,
        )

        return model_training_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            X_test_dir=config.X_test_dir,
            y_test_dir=config.y_test_dir,
            preprocessor_dir=config.preprocessor_dir,
            model_dir=config.model_dir,
            all_params=self.params,
            target_column=schema.name,
            mlflow_uri=os.getenv("MLFLOW_TRACKING_URI"),
        )
        return model_evaluation_config
