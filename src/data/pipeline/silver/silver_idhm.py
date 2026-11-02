import logging
import re
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

FOLDER = "idhm"
FILE_NAME = "adh_radar_base_2012_2024.xlsx"
SHEET_NAME = "Total"
YEARS = (2022, 2024)
AGREGACAO_UF = "Unidade da Federação"
COLUMNS_REGEX = r'^(IDHM|IDHM_E|IDHM_L|IDHM_R|I_ESCOLARIDADE|I_FREQ_PROP|T_FREQ.*|T_FUND.*|T_MED.*|T_ANALF.*|T_ATRASO_.*|ANOSEST|RDPC|RDPCT|RDPC\d+|GINI|THEIL)$'

# nomes originais do Radar para inglês em minúsculo, igual ao notebook
RENAME = {
    'IDHM': 'hdi',
    'IDHM_L': 'hdi_life_expectancy',
    'IDHM_E': 'hdi_education',
    'IDHM_R': 'hdi_income',
    'I_ESCOLARIDADE': 'education_subindex',
    'I_FREQ_PROP': 'school_attendance_subindex',
    'T_FREQ5A6': 'school_attendance_rate_5_6',
    'T_FREQ6A14': 'school_attendance_rate_6_14',
    'T_FREQ15A17': 'school_attendance_rate_15_17',
    'T_FREQ18A24': 'school_attendance_rate_18_24',
    'T_FUND11A13': 'elementary_completion_rate_11_13',
    'T_FUND15A17': 'elementary_completion_rate_15_17',
    'T_FUND18A24': 'elementary_completion_rate_18_24',
    'T_FUND18M': 'elementary_completion_rate_18_plus',
    'T_FUND25M': 'elementary_completion_rate_25_plus',
    'T_MED18A20': 'high_school_completion_rate_18_20',
    'T_MED25M': 'high_school_completion_rate_25_plus',
    'T_ANALF15M': 'illiteracy_rate_15_plus',
    'T_ANALF18M': 'illiteracy_rate_18_plus',
    'T_ANALF25M': 'illiteracy_rate_25_plus',
    'T_ATRASO_2_BASICO': 'age_lag_2y_basic_rate',
    'T_ATRASO_2_FUND': 'age_lag_2y_elementary_rate',
    'ANOSEST': 'expected_years_of_schooling',
    'RDPC': 'income_per_capita_mean',
    'RDPCT': 'income_per_capita_nonzero_mean',
    'RDPC1': 'income_per_capita_quintile_1',
    'RDPC2': 'income_per_capita_quintile_2',
    'RDPC3': 'income_per_capita_quintile_3',
    'RDPC4': 'income_per_capita_quintile_4',
    'RDPC5': 'income_per_capita_quintile_5',
    'RDPC10': 'income_per_capita_decile_10',
    'GINI': 'gini_index',
    'THEIL': 'theil_index',
}


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

def read_radar(spark, uri: str) -> DataFrame:
    """Baixa o XLSX do S3 e lê a aba Total como texto, pois o Spark não lê Excel."""

    parsed = urlparse(uri)
    if parsed.scheme == "file":
        pdf = pd.read_excel(parsed.path, sheet_name=SHEET_NAME, dtype=str)
        pdf = pdf.where(pdf.notna(), None)
        return spark.createDataFrame(pdf)

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / Path(parsed.path).name
        boto3.client("s3").download_file(parsed.netloc, parsed.path.lstrip("/"), str(local_path))
        pdf = pd.read_excel(local_path, sheet_name=SHEET_NAME, dtype=str)
    pdf = pdf.where(pdf.notna(), None)
    return spark.createDataFrame(pdf)


def transform_idhm(dataframe: DataFrame) -> DataFrame:
    """Filtra as UFs de 2022 a 2024, seleciona as colunas pedidas, renomeia e converte os tipos."""

    dataframe = dataframe.filter(
        (F.col("AGREGACAO") == AGREGACAO_UF) & F.col("ANO").cast("double").between(*YEARS)
    )
    colunas = [coluna for coluna in dataframe.columns if re.match(COLUMNS_REGEX, coluna)]
    dataframe = dataframe.select(
        F.col("ANO").cast("double").cast("int").alias("year"),
        F.col("CODIGO").cast("double").cast("int").alias("state_code"),
        F.col("NOME").alias("state_name"),
        *[F.col(coluna).cast("double").alias(RENAME[coluna]) for coluna in colunas],
    )
    return dataframe.orderBy("year", "state_code")


def load(dataframe: DataFrame, bucket_silver: str, folder: str) -> str:
    """Grava o resultado em Parquet na Silver, particionado por ano."""

    path = f"{bucket_silver}/{folder}/idhm_uf"
    dataframe.write.mode("overwrite").partitionBy("year").parquet(path)
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

uri = f"{BUCKET_BRONZE}/{FOLDER}/{FILE_NAME}"
dataframe = read_radar(spark, uri)
dataframe_uf = transform_idhm(dataframe)
load_path = load(dataframe_uf, BUCKET_SILVER, FOLDER)
log.info(f"Normalizado {uri} -> {load_path}")

job.commit()
