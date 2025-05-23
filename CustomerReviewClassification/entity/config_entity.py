from dataclasses import dataclass
from pathlib import Path


@dataclass
class ETL_Config:
    raw_data_dir: Path
    table_name: str


@dataclass
class DataIngestionConfig:
    root_dir: Path
    ingested_dir: Path
    table_name: str


@dataclass
class DataValidationConfig:
    root_dir: Path
    train_dir: Path
    test_dir: Path
    status_dir: Path
    all_schema: dict
    oov_threshold: float


@dataclass
class DataTransformationConfig:
    root_dir: Path
    train_dir: Path
    test_dir: Path
    preprocessor_dir: Path


@dataclass
class ModelTrainingConfig:
    root_dir: Path
    X_train_dir: Path
    y_train_dir: Path
    params: dict


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    X_test_dir: Path
    y_test_dir: Path
    preprocessor_dir: Path
    model_dir: Path
    all_params: dict
    target_column: str
    mlflow_uri: str


@dataclass
class ModelPredictionConfig:
    model_dir: Path
    preprocessor_dir: Path
