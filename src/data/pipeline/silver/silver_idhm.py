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

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'BUCKET_BRONZE', 'BUCKET_SILVER'])  # lê os parâmetros passados ao job
JOB_NAME = args['JOB_NAME']  # nome do job, exigido pelo Glue
BUCKET_BRONZE = args['BUCKET_BRONZE']  # origem: camada Bronze no S3
BUCKET_SILVER = args['BUCKET_SILVER']  # destino: camada Silver no S3


# ============================================================
# VARIABLES
# ============================================================

FOLDER = "idhm"  # pasta da Bronze processada por este job
FILE_NAME = "adh_radar_base_2012_2024.xlsx"  # arquivo do Radar usado pelo notebook
SHEET_NAME = "Total"  # aba com uma linha por território e ano
YEARS = (2022, 2024)  # recorte de anos, igual ao notebook
AGREGACAO_UF = "Unidade da Federação"  # filtra só as Unidades da Federação
COLUMNS_REGEX = r'^(IDHM|IDHM_E|IDHM_L|IDHM_R|I_ESCOLARIDADE|I_FREQ_PROP|T_FREQ.*|T_FUND.*|T_MED.*|T_ANALF.*|T_ATRASO_.*|ANOSEST|RDPC|RDPCT|RDPC\d+|GINI|THEIL)$'  # colunas pedidas, igual ao notebook

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
)  # formato de log legível no CloudWatch
log = logging.getLogger(__name__)  # logger do script


# ============================================================
# FUNCTIONS
# ============================================================

def read_radar(spark, uri: str) -> DataFrame:
    """Baixa o XLSX do S3 e lê a aba Total como texto, pois o Spark não lê Excel."""

    parsed = urlparse(uri)  # separa o bucket e a chave da URI
    with tempfile.TemporaryDirectory() as temp_dir:  # diretório temporário local, removido ao final
        local_path = Path(temp_dir) / Path(parsed.path).name  # caminho local do XLSX
        boto3.client("s3").download_file(parsed.netloc, parsed.path.lstrip("/"), str(local_path))  # baixa o arquivo
        pdf = pd.read_excel(local_path, sheet_name=SHEET_NAME, dtype=str)  # lê a aba Total como texto
    pdf = pdf.where(pdf.notna(), None)  # troca NaN por nulo para o Spark aceitar
    return spark.createDataFrame(pdf)  # converte para DataFrame Spark


def transform_idhm(dataframe: DataFrame) -> DataFrame:
    """Filtra as UFs de 2022 a 2024, seleciona as colunas pedidas, renomeia e converte os tipos."""

    dataframe = dataframe.filter(
        (F.col("AGREGACAO") == AGREGACAO_UF) & F.col("ANO").cast("double").between(*YEARS)  # só UFs dentro do recorte
    )
    colunas = [coluna for coluna in dataframe.columns if re.match(COLUMNS_REGEX, coluna)]  # colunas que casam com o padrão
    dataframe = dataframe.select(
        F.col("ANO").cast("double").cast("int").alias("year"),  # ano como inteiro
        F.col("CODIGO").cast("double").cast("int").alias("state_code"),  # código IBGE da UF como inteiro
        F.col("NOME").alias("state_name"),  # nome da UF
        *[F.col(coluna).cast("double").alias(RENAME[coluna]) for coluna in colunas],  # indicadores como número, com nome em inglês
    )
    return dataframe.orderBy("year", "state_code")  # ordena por ano e código da UF


def load(dataframe: DataFrame, bucket_silver: str, folder: str) -> str:
    """Grava o resultado em Parquet na Silver, particionado por ano."""

    path = f"{bucket_silver}/{folder}/idhm_uf"  # destino: silver/idhm/idhm_uf
    dataframe.write.mode("overwrite").partitionBy("year").parquet(path)  # uma pasta por ano
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

sc = SparkContext()  # contexto Spark exigido pelo GlueContext
glueContext = GlueContext(sc)  # contexto do Glue
spark = glueContext.spark_session  # sessão Spark para transformação
job = Job(glueContext)  # objeto que controla o ciclo de vida do job
job.init(JOB_NAME, args)  # inicia o job com os parâmetros resolvidos

uri = f"{BUCKET_BRONZE}/{FOLDER}/{FILE_NAME}"  # arquivo de origem na Bronze
dataframe = read_radar(spark, uri)  # lê a aba Total
dataframe_uf = transform_idhm(dataframe)  # filtra, seleciona e renomeia
load_path = load(dataframe_uf, BUCKET_SILVER, FOLDER)  # grava na Silver particionado por ano
log.info(f"Normalizado {uri} -> {load_path}")

job.commit()  # finaliza o job e registra o estado no Glue
