from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.entity.config_entity import SGD_ModelTrainingConfig
from CustomerReviewClassification.utils.common import save_bin
from sklearn.linear_model import SGDClassifier
from scipy.sparse import load_npz
import pandas as pd


class SGD_ModelTraining:
    def __init__(self, sgd_model_training_config: SGD_ModelTrainingConfig):
        self.config = sgd_model_training_config

    def train(self):
        X_train = load_npz(self.config.X_train_dir)
        y_train = pd.read_csv(self.config.y_train_dir)
        logging.info("Loaded X_train and y_train successfully")

        sgd = SGDClassifier(
            loss=self.config.loss,
            penalty=self.config.penalty,
            max_iter=self.config.max_iter,
        )

        logging.info("The SGD fitting process has started")
        sgd.fit(X_train, y_train.values.ravel())
        logging.info("The SGD fitting process has finished")

        save_bin(data=sgd, file_path=self.config.model_dir)
