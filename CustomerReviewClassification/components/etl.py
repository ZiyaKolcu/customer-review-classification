import os
import sys
import pandas as pd
import psycopg2
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.entity.config_entity import ETL_Config
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

PG_TYPE_MAPPING = {
    "int64": "INTEGER",
    "float64": "DOUBLE PRECISION",
    "object": "TEXT",
}


class ETL:
    def __init__(self, etl_config: ETL_Config):
        self.etl_config = etl_config

        try:
            self.conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST"),
                port=os.getenv("POSTGRES_PORT"),
                database=os.getenv("POSTGRES_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
            )
            self.cur = self.conn.cursor()
        except Exception as e:
            raise CustomException(e, sys)

    def create_table_if_not_exists(self, df):
        column_defs = []
        for col, dtype in df.dtypes.items():
            pg_type = PG_TYPE_MAPPING.get(str(dtype), "TEXT")
            column_defs.append(f'"{col}" {pg_type}')

        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.etl_config.table_name} (
            {", ".join(column_defs)}
        );
        """
        self.cur.execute(create_table_query)
        self.conn.commit()

    def etl(self):
        df = pd.read_csv(self.etl_config.raw_data_dir)

        df["label"] = df["label"].replace({1: 0, 2: 1})

        self.create_table_if_not_exists(df)

        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False, header=False)
        csv_buffer.seek(0)

        sql_query = f"""
           COPY {self.etl_config.table_name} 
           FROM stdin 
           WITH CSV HEADER 
           DELIMITER as ',' 
        """

        self.cur.copy_expert(sql=sql_query, file=csv_buffer)

        self.conn.commit()
        self.cur.close()
        self.conn.close()
