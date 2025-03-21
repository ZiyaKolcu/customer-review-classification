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
