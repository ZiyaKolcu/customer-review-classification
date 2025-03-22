import os
import pandas as pd
from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.entity.config_entity import DataTransformationConfig
from CustomerReviewClassification.utils.common import save_bin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import save_npz


class DataTransformation:
    def __init__(self, data_transform_config: DataTransformationConfig):
        self.config = data_transform_config

    def get_preprocessor(self):
        return Pipeline(
            [
                ("vectorizer", CountVectorizer()),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )

    def transform(self):
        train_data = pd.read_csv(self.config.train_dir)
        test_data = pd.read_csv(self.config.test_dir)

        X_train = train_data.iloc[:, 1:]
        X_test = test_data.iloc[:, 1:]

        y_train = train_data.iloc[:, 0]
        y_test = test_data.iloc[:, 0]

        preprocessor = self.get_preprocessor()
        save_bin(data=preprocessor, file_path=self.config.preprocessor_dir)

        X_train_processed = preprocessor.fit_transform(
            X_train.astype(str).values.flatten()
        )
        logging.info("X_train transformed")

        train_dir = os.path.join(self.config.root_dir, "X_train.npz")
        save_npz(train_dir, X_train_processed)
        logging.info("X_train.npz saved")
        X_test.to_csv(os.path.join(self.config.root_dir, "X_test.csv"), index=False)
        logging.info("X_test.csv saved")
        y_train.to_csv(os.path.join(self.config.root_dir, "y_train.csv"), index=False)
        logging.info("y_train.csv saved")
        y_test.to_csv(os.path.join(self.config.root_dir, "y_test.csv"), index=False)
        logging.info("y_test.csv saved")
