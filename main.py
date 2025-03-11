import sys
from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.pipeline.etl_pipeline import ETL_Pipeline

if __name__ == "__main__":
    try:
        logging.info(f">>>>> ETL pipeline started <<<<<")
        etl_pipeline = ETL_Pipeline()
        etl_pipeline.initiate_etl()
        logging.info(">>>>> ETL pipeline completed <<<<<")

    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)
