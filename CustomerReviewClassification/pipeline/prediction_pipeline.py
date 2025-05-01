from CustomerReviewClassification.configuration.configuration import (
    ConfigurationManager,
)
from CustomerReviewClassification.components.model_prediction import ModelPrediction
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.logging.logger import logging


class PredictionPipeline:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config_manager = ConfigurationManager()

    def initiate_model_prediction(self, review_text: str):
        model_prediction_config = self.config_manager.get_model_prediction_config()
        model_prediction = ModelPrediction(
            model_name=self.model_name, model_prediction_config=model_prediction_config
        )
        return model_prediction.predict(review_text=review_text)
