import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from CustomerReviewClassification.logging.logger import logging
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.entity.config_entity import ModelEvaluationConfig
from CustomerReviewClassification.utils.common import load_bin, save_json
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


class ModelEvaluation:
    def __init__(self, model_name, model_evaluation_config: ModelEvaluationConfig):
        self.model_name = model_name
        self.config = model_evaluation_config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return accuracy, precision, recall, f1

    def load_data(self):
        X_test = pd.read_csv(self.config.X_test_dir)
        y_test = pd.read_csv(self.config.y_test_dir).values.ravel()
        logging.info("Loaded X_test and y_test")
        return X_test, y_test

    def load_preprocessor(self):
        preprocessor = load_bin(self.config.preprocessor_dir)
        logging.info("Loaded preprocessor")
        return preprocessor

    def load_model(self):
        model = load_bin(
            self.config.model_dir + "/" + self.model_name + "_model.joblib"
        )

        logging.info("Loaded ML model")
        return model

    def log_into_mlflow(self):
        X_test, y_test = self.load_data()
        preprocessor = self.load_preprocessor()
        model = self.load_model()

        X_test_processed = preprocessor.transform(X_test.astype(str).values.flatten())
        logging.info("Transformed X_test")

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{self.model_name}_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            predicted_labels = model.predict(X_test_processed)
            accuracy, precision, recall, f1 = self.eval_metrics(
                y_test, predicted_labels
            )

            scores = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            save_json(
                path=Path(
                    self.config.root_dir + "/" + self.model_name + "_metrics.json"
                ),
                data=scores,
            )

            mlflow.log_params(self.config.all_params[self.model_name])

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model,
                    self.model_name + "_model",
                    registered_model_name=self.model_name,
                )
            else:
                mlflow.sklearn.log_model(model, self.model_name)
