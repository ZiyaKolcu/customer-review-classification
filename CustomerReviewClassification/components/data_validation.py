import os
import sys
import json
import pandas as pd
from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.entity.config_entity import DataValidationConfig
from sklearn.feature_extraction.text import CountVectorizer


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig):
        self.config = data_validation_config

    def load_data(self):
        train_data = pd.read_csv(self.config.train_dir)
        test_data = pd.read_csv(self.config.test_dir)

        return train_data, test_data

    def compute_oov_ratio(self, train_data, test_data):
        vectorizer = CountVectorizer()
        vectorizer.fit(train_data)
        train_vocab = set(vectorizer.get_feature_names_out())

        test_vectorizer = CountVectorizer()
        test_vectorizer.fit(test_data)
        test_vocab = set(test_vectorizer.get_feature_names_out())

        new_words = test_vocab - train_vocab
        oov_ratio = len(new_words) / len(test_vocab)
        return oov_ratio

    def validate_oov(self) -> bool:
        try:
            logging.info("Started Validate OOV")
            validation_status = None

            train_data, test_data = self.load_data()

            oov_ratio = self.compute_oov_ratio(train_data, test_data)

            validation_status = oov_ratio <= self.config.oov_threshold

            status_data = {"Validation Status": validation_status}
            logging.info(f"Validation Status: {status_data}")

            with open(self.config.status_dir, "w") as f:
                json.dump(status_data, f, indent=4)
                logging.info(f"Validation status file saved at {self.config.root_dir}")

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)
