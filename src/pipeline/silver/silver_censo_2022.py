"""Normaliza os dados raciais municipais do Censo 2022 em Parquet."""

import logging
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext


args = getResolvedOptions(sys.argv, ["JOB_NAME", "BUCKET_BRONZE", "BUCKET_SILVER"])
JOB_NAME = args["JOB_NAME"]
BUCKET_BRONZE = args["BUCKET_BRONZE"]
BUCKET_SILVER = args["BUCKET_SILVER"]

FOLDER = "censo_2022"
FILE_NAME = "dados_raciais_2022_MU.xlsx"
MUNICIPAL_FILE_NAME = "censo_2022_dados_municipais.csv"
OUTPUT_PATH = "censo_2022"
RACIAL_COLUMNS = [
    "city_name",
    "city_code",
    "population_total",
    "population_white",
    "population_black",
    "population_yellow",
    "population_brown",
    "population_indigenous",
]
MUNICIPAL_COLUMNS = {
    "id_municipio": "city_code",
    "sigla_uf": "state_abbreviation",
    "domicilios": "households",
    "area": "area",
    "taxa_alfabetizacao": "literacy_rate",
    "idade_mediana": "median_age",
    "razao_sexo": "sex_ratio",
    "indice_envelhecimento": "aging_index",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)


def create_s3_client() -> object:
    """Cria um cliente S3 e aceita um endpoint MinIO em testes locais."""

    endpoint = os.getenv("LOCAL_S3_ENDPOINT")
    if endpoint:
        return boto3.client("s3", endpoint_url=endpoint)
    return boto3.client("s3")


def read_excel_source(uri: str) -> pd.DataFrame:
    """Lê a planilha do Excel, localmente ou a partir do S3."""

    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return pd.read_excel(parsed.path, skiprows=8, header=None)

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / Path(parsed.path).name
        create_s3_client().download_file(
            parsed.netloc,
            parsed.path.lstrip("/"),
            str(local_path),
        )
        return pd.read_excel(local_path, skiprows=8, header=None)


def read_csv_source(uri: str) -> pd.DataFrame:
    """Lê o CSV municipal, localmente ou a partir do S3."""

    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return pd.read_csv(parsed.path, sep=",", encoding="latin-1")

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / Path(parsed.path).name
        create_s3_client().download_file(
            parsed.netloc,
            parsed.path.lstrip("/"),
            str(local_path),
        )
        return pd.read_csv(local_path, sep=",", encoding="latin-1")


def normalize_censo(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aplica as mesmas seleções, renomeações e conversões do notebook."""

    normalized = dataframe.iloc[:, :8].copy()
    normalized.columns = RACIAL_COLUMNS

    normalized["city_code"] = pd.to_numeric(
        normalized["city_code"],
        errors="coerce",
    )
    normalized = normalized.dropna(subset=["city_code"]).copy()

    for column in RACIAL_COLUMNS[1:]:
        normalized[column] = (
            normalized[column]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace("-", "0", regex=False)
        )
        normalized[column] = (
            pd.to_numeric(normalized[column], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    normalized["city_name"] = normalized["city_name"].astype("string")
    return normalized


def normalize_municipal(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Seleciona e renomeia os indicadores municipais usados pelo notebook."""

    columns = [column for column in MUNICIPAL_COLUMNS if column in dataframe.columns]
    normalized = dataframe[columns].rename(columns=MUNICIPAL_COLUMNS).copy()
    normalized["city_code"] = pd.to_numeric(
        normalized["city_code"],
        errors="coerce",
    )
    normalized = normalized.dropna(subset=["city_code"]).copy()
    normalized["city_code"] = normalized["city_code"].astype("int32")

    numeric_columns = [
        "households",
        "area",
        "literacy_rate",
        "median_age",
        "sex_ratio",
        "aging_index",
    ]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    return normalized


def build_censo_2022(
    racial_data: pd.DataFrame,
    municipal_data: pd.DataFrame,
) -> pd.DataFrame:
    """Combina os dados municipais e raciais como no notebook, pelo código IBGE."""

    racial = normalize_censo(racial_data)
    municipal = normalize_municipal(municipal_data)
    return municipal.merge(racial, on="city_code", how="left")


def load_parquet(dataframe: pd.DataFrame, spark: object, uri: str) -> str:
    """Converte o resultado Pandas para Spark e o grava em Parquet na Silver."""

    spark_dataframe = spark.createDataFrame(dataframe)
    spark_dataframe.write.mode("overwrite").parquet(uri)
    return uri


sc = SparkContext()
glue_context = GlueContext(sc)
job = Job(glue_context)
job.init(JOB_NAME, args)

input_uri = f"{BUCKET_BRONZE}/{FOLDER}/{FILE_NAME}"
municipal_input_uri = f"{BUCKET_BRONZE}/{FOLDER}/{MUNICIPAL_FILE_NAME}"
output_uri = f"{BUCKET_SILVER}/{FOLDER}/{OUTPUT_PATH}"

log.info("Lendo %s", input_uri)
log.info("Lendo %s", municipal_input_uri)
raw_racial_data = read_excel_source(input_uri)
raw_municipal_data = read_csv_source(municipal_input_uri)
normalized_data = build_censo_2022(raw_racial_data, raw_municipal_data)
load_parquet(normalized_data, glue_context.spark_session, output_uri)
log.info("Arquivo normalizado: %s (%d municípios)", output_uri, len(normalized_data))

job.commit()
