import sys
from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.pipeline.etl_pipeline import ETL_Pipeline
from CustomerReviewClassification.pipeline.svm_training_pipeline import (
    SVM_Training_Pipeline,
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
        logging.info(f">>>>> SVM Training pipeline started <<<<<")
        svm_training_pipeline = SVM_Training_Pipeline()
        svm_training_pipeline.initiate_data_ingestion()
        logging.info(">>>>> SVM Training pipeline completed <<<<<")

    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
