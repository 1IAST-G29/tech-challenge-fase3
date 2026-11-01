"""Normaliza os dados raciais municipais do Censo 2022.

Este job reproduz a transformação do notebook ``censo_2022.ipynb`` e gera o
arquivo ``censo_2022/censo_2022.csv`` na camada Silver.
"""

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
OUTPUT_NAME = "censo_2022.csv"
OUTPUT_COLUMNS = [
    "city_name",
    "city_code",
    "population_total",
    "population_white",
    "population_black",
    "population_yellow",
    "population_brown",
    "population_indigenous",
]
INTEGER_COLUMNS = OUTPUT_COLUMNS[1:]

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


def normalize_censo(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aplica as mesmas seleções, renomeações e conversões do notebook."""

    normalized = dataframe.iloc[:, :8].copy()
    normalized.columns = OUTPUT_COLUMNS

    normalized["city_code"] = pd.to_numeric(
        normalized["city_code"],
        errors="coerce",
    )
    normalized = normalized.dropna(subset=["city_code"]).copy()

    for column in INTEGER_COLUMNS:
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


def load_csv(dataframe: pd.DataFrame, uri: str) -> str:
    """Grava o CSV UTF-8 separado por ponto e vírgula na camada Silver."""

    parsed = urlparse(uri)
    if parsed.scheme == "file":
        output_path = Path(parsed.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False, sep=";", encoding="utf-8")
        return uri

    with tempfile.NamedTemporaryFile(suffix=".csv") as temporary_file:
        dataframe.to_csv(
            temporary_file.name,
            index=False,
            sep=";",
            encoding="utf-8",
        )
        create_s3_client().upload_file(
            temporary_file.name,
            parsed.netloc,
            parsed.path.lstrip("/"),
        )
    return uri


sc = SparkContext()
glue_context = GlueContext(sc)
job = Job(glue_context)
job.init(JOB_NAME, args)

input_uri = f"{BUCKET_BRONZE}/{FOLDER}/{FILE_NAME}"
output_uri = f"{BUCKET_SILVER}/{FOLDER}/{OUTPUT_NAME}"

log.info("Lendo %s", input_uri)
raw_data = read_excel_source(input_uri)
normalized_data = normalize_censo(raw_data)
load_csv(normalized_data, output_uri)
log.info("Arquivo normalizado: %s (%d municípios)", output_uri, len(normalized_data))

job.commit()
