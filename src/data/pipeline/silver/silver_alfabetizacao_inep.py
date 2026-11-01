import logging
import re
import sys
import unicodedata
from functools import reduce
from pathlib import Path
from urllib.parse import urlparse

import boto3
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# ============================================================
# JOB PARAMETERS
# ============================================================
#
#   --BUCKET_BRONZE s3://bucket/bronze
#   --BUCKET_SILVER s3://bucket/silver

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'BUCKET_BRONZE', 'BUCKET_SILVER'])
JOB_NAME = args['JOB_NAME']
BUCKET_BRONZE = args['BUCKET_BRONZE']
BUCKET_SILVER = args['BUCKET_SILVER']


# ============================================================
# VARIABLES
# ============================================================

FOLDER = "alfabetizacao_inep"
CSV_SEPARATOR = ";"
CSV_ENCODING = "ISO-8859-1"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)


# ============================================================
# FUNCTIONS
# ============================================================

def list_files(bucket_uri: str, folder: str) -> list[str]:
    """Lista as URIs de todos os arquivos de uma pasta da Bronze."""

    parsed = urlparse(bucket_uri)
    if parsed.scheme == "file":
        return [
            str(path)
            for path in (Path(parsed.path) / folder).rglob("*")
            if path.is_file()
        ]

    prefix = f"{parsed.path.strip('/')}/{folder}/".lstrip("/")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    uris = []
    for page in paginator.paginate(Bucket=parsed.netloc, Prefix=prefix):
        for item in page.get("Contents", []):
            uris.append(f"s3://{parsed.netloc}/{item['Key']}")
    return uris


def normalize_column(name: str) -> str:
    """Converte o nome de uma coluna para snake_case, sem acentos ou caracteres especiais."""

    sem_acento = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", sem_acento.lower()).strip("_")


def normalize(dataframe: DataFrame, source_file: str) -> DataFrame:
    """Padroniza nomes de colunas, remove espaços extras e valores vazios e elimina duplicados."""

    dataframe = dataframe.toDF(*[normalize_column(column) for column in dataframe.columns])
    dataframe = dataframe.select([
        F.when(F.trim(F.col(column)) == "", None).otherwise(F.trim(F.col(column))).alias(column)
        for column in dataframe.columns
    ])
    dataframe = dataframe.dropDuplicates()
    dataframe = (dataframe
                    .withColumn("_source_file", F.lit(source_file))
                    .withColumn("_ingested_at", F.current_timestamp())
                )
    return dataframe


def load(dataframe: DataFrame, bucket_silver: str, folder: str, entity: str) -> str:
    """Grava a entidade concatenada em Parquet na Silver, particionada por ano."""

    path = f"{bucket_silver}/{folder}/{entity}"
    dataframe.write.mode("overwrite").partitionBy("ano").parquet(path)
    return path


# ============================================================
# EXECUTION
# ============================================================

log.info("=" * 60)
log.info(f"JOB_NAME : {JOB_NAME}")
log.info(f"FOLDER : {FOLDER}")
log.info(f"BUCKET_BRONZE : {BUCKET_BRONZE}")
log.info(f"BUCKET_SILVER : {BUCKET_SILVER}")
log.info("=" * 60)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(JOB_NAME, args)

entities = {}

for uri in list_files(BUCKET_BRONZE, FOLDER):
    file_name = Path(uri).name
    if not file_name.lower().endswith(".csv"):
        log.warning(f"Arquivo ignorado (formato não suportado): {uri}")
        continue
    ano, entity = Path(file_name).stem.split("_", 1)
    dataframe = (spark.read
                    .option("header", True)
                    .option("sep", CSV_SEPARATOR)
                    .option("encoding", CSV_ENCODING)
                    .csv(uri)
                )
    dataframe_normalized = normalize(dataframe, uri).withColumn("ano", F.lit(int(ano)))
    entities.setdefault(entity, []).append(dataframe_normalized)

for entity, dataframes in entities.items():
    dataframe = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dataframes)
    load_path = load(dataframe, BUCKET_SILVER, FOLDER, entity)
    log.info(f"Entidade {entity}: {len(dataframes)} arquivos -> {load_path}")

job.commit()
