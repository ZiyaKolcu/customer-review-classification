import sys
from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.pipeline.etl_pipeline import ETL_Pipeline
from CustomerReviewClassification.pipeline.lr_training_pipeline import (
    LR_Training_Pipeline,
)

# if __name__ == "__main__":
#     try:
#         logging.info(f">>>>> ETL pipeline started <<<<<")
#         etl_pipeline = ETL_Pipeline()
#         etl_pipeline.initiate_etl()
#         logging.info(">>>>> ETL pipeline completed <<<<<")

#     except Exception as e:
#         logging.exception(e)
#         raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        logging.info(f">>>>> LR Training pipeline started <<<<<")
        lr_training_pipeline = LR_Training_Pipeline()
        # lr_training_pipeline.initiate_data_ingestion()
        # lr_training_pipeline.initiate_data_validation()
        lr_training_pipeline.initiate_data_transform()
        # lr_training_pipeline.initiate_lr_model_training()
        lr_training_pipeline.initiate_model_evaluation()
        logging.info(">>>>> LR Training pipeline completed <<<<<")

    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
