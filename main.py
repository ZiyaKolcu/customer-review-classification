import sys
from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.pipeline.etl_pipeline import ETL_Pipeline
from CustomerReviewClassification.pipeline.training_pipeline import Training_Pipeline


if __name__ == "__main__":
    try:
        # logging.info(f">>>>> ETL pipeline started <<<<<")
        # etl_pipeline = ETL_Pipeline()
        # etl_pipeline.initiate_etl()
        # logging.info(">>>>> ETL pipeline completed <<<<<")
        logging.info(f">>>>> LogisticRegression Training pipeline started <<<<<")
        lr_training_pipeline = Training_Pipeline(model_name="LogisticRegression")
        lr_training_pipeline.initiate_data_ingestion()
        lr_training_pipeline.initiate_data_validation()
        lr_training_pipeline.initiate_data_transform()
        lr_training_pipeline.initiate_model_training()
        lr_training_pipeline.initiate_model_evaluation()
        logging.info(">>>>> LogisticRegression Training pipeline completed <<<<<")
        logging.info(
            f">>>>> GradientBoostingClassifier Training pipeline started <<<<<"
        )
        gbc_training_pipeline = Training_Pipeline(
            model_name="GradientBoostingClassifier"
        )
        gbc_training_pipeline.initiate_data_ingestion()
        gbc_training_pipeline.initiate_data_validation()
        gbc_training_pipeline.initiate_data_transform()
        gbc_training_pipeline.initiate_model_training()
        gbc_training_pipeline.initiate_model_evaluation()
        logging.info(
            ">>>>> GradientBoostingClassifier Training pipeline completed <<<<<"
        )
        logging.info(f">>>>> SGDClassifier Training pipeline started <<<<<")
        sgd_training_pipeline = Training_Pipeline(model_name="SGDClassifier")
        sgd_training_pipeline.initiate_data_ingestion()
        sgd_training_pipeline.initiate_data_validation()
        sgd_training_pipeline.initiate_data_transform()
        sgd_training_pipeline.initiate_model_training()
        sgd_training_pipeline.initiate_model_evaluation()
        logging.info(">>>>> SGDClassifier Training pipeline completed <<<<<")

    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
