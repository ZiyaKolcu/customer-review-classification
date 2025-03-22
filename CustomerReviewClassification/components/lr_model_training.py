from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.entity.config_entity import LR_ModelTrainingConfig
from CustomerReviewClassification.utils.common import save_bin
from sklearn.linear_model import LogisticRegression
from scipy.sparse import load_npz
import pandas as pd


class LR_ModelTraining:
    def __init__(self, lr_model_training_config: LR_ModelTrainingConfig):
        self.config = lr_model_training_config

    def train(self):
        X_train = load_npz(self.config.X_train_dir)
        y_train = pd.read_csv(self.config.y_train_dir)
        logging.info("Loaded X_train and y_train successfully")

        lr = LogisticRegression(
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            C=self.config.C,
        )

        logging.info("The LR fitting process has started")
        lr.fit(X_train, y_train.values.ravel())
        logging.info("The LR fitting process has finished")

        save_bin(data=lr, file_path=self.config.model_dir)
