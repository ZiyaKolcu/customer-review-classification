from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.entity.config_entity import ModelPredictionConfig
from CustomerReviewClassification.utils.common import load_bin


class ModelPrediction:
    def __init__(self, model_name: str, model_prediction_config: ModelPredictionConfig):
        self.model_name = model_name
        self.config = model_prediction_config

    def load_model(self):
        model = load_bin(
            self.config.model_dir + "/" + self.model_name + "_model.joblib"
        )

        logging.info("Loaded ML model")
        return model

    def load_preprocessor(self):
        preprocessor = load_bin(self.config.preprocessor_dir)
        logging.info("Loaded preprocessor")
        return preprocessor

    def predict(self, review_text):
        model = self.load_model()
        preprocessor = self.load_preprocessor()

        if isinstance(review_text, str):
            review_text = [review_text]  

        processed_text = preprocessor.transform(review_text)

        prediction = model.predict(processed_text)

        return prediction[0]

