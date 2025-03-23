from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.entity.config_entity import GBC_ModelTrainingConfig
from CustomerReviewClassification.utils.common import save_bin
from sklearn.ensemble import GradientBoostingClassifier
from scipy.sparse import load_npz
import pandas as pd


class GBC_ModelTraining:
    def __init__(self, gbc_model_training_config: GBC_ModelTrainingConfig):
        self.config = gbc_model_training_config

    def train(self):
        X_train = load_npz(self.config.X_train_dir)
        y_train = pd.read_csv(self.config.y_train_dir)
        logging.info("Loaded X_train and y_train successfully")

        gbc = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
        )

        logging.info("The GBC fitting process has started")
        gbc.fit(X_train, y_train.values.ravel())
        logging.info("The GBC fitting process has finished")

        save_bin(data=gbc, file_path=self.config.model_dir)
