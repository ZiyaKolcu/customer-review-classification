from dataclasses import dataclass
from pathlib import Path


@dataclass
class ETL_Config:
    raw_data_dir: Path
    table_name: str
