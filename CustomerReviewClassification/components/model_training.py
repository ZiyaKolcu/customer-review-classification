from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.entity.config_entity import ModelTrainingConfig
from CustomerReviewClassification.utils.common import save_bin
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from scipy.sparse import load_npz
import pandas as pd


class ModelTraining:
    def __init__(self, model_training_config: ModelTrainingConfig):
        self.config = model_training_config
        self.lr_params = self.config.params.LogisticRegression
        self.gbc_params = self.config.params.GradientBoostingClassifier
        self.sgd_params = self.config.params.SGDClassifier

    def load_data(self):
        X_train = load_npz(self.config.X_train_dir)
        y_train = pd.read_csv(self.config.y_train_dir)
        logging.info("Loaded X_train and y_train successfully")

        return X_train, y_train

    def get_model(self, model_name: str):
        if model_name == "LR":
            model = LogisticRegression(
                solver=self.lr_params.solver,
                max_iter=self.lr_params.max_iter,
                class_weight=self.lr_params.class_weight,
                C=self.lr_params.C,
            )
        elif model_name == "GBC":
            model = GradientBoostingClassifier(
                n_estimators=self.gbc_params.n_estimators,
                learning_rate=self.gbc_params.learning_rate,
                max_depth=self.gbc_params.max_depth,
                random_state=self.gbc_params.random_state,
            )
        elif model_name == "SGD":
            model = SGDClassifier(
                loss=self.sgd_params.loss,
                penalty=self.sgd_params.penalty,
                max_iter=self.sgd_params.max_iter,
            )
        return model

    def train(self, model_name: str):
        X_train, y_train = self.load_data()

        model = self.get_model(model_name)

        model.fit(X_train, y_train.values.ravel())

        save_bin(
            data=model, file_path=self.config.root_dir + f"/{model_name}_model.joblib"
        )
