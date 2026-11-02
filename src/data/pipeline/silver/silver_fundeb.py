import logging
import sys

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

FOLDER = "fundeb"
FILE_NAME = "fundeb-por-municipio.csv"
CSV_SEPARATOR = ";"
CSV_ENCODING = "ISO-8859-1"
# Mantemos somente os anos usados pela base de modelagem.
YEARS = ["2023", "2024", "2025"]


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

def read_fundeb(spark, uri: str) -> DataFrame:
    """Lê o CSV e mantém os campos padronizados e os anos de 2023 a 2025."""

    dataframe = (spark.read
                    .option("header", True)
                    .option("sep", CSV_SEPARATOR)
                    .option("encoding", CSV_ENCODING)
                    .csv(uri)
                )
    dataframe = dataframe.select(
        F.col("COD_MUN").cast("int").alias("city_code"),
        F.trim(F.col("Município")).cast("string").alias("city_name"),
        F.trim(F.col("UF")).cast("string").alias("state_abbreviation"),
        *[F.col(ano) for ano in YEARS],
    )
    return dataframe.filter(F.col("city_name").isNotNull())


def parse_amount(column: str):
    """Converte o texto no formato brasileiro (1.926.974,37) para número; '-' e vazio viram nulo."""

    texto = F.regexp_replace(F.regexp_replace(F.trim(F.col(column)), r"\.", ""), ",", ".")
    return F.when(texto.rlike(r"^-?[0-9]+(\.[0-9]+)?$"), texto.cast("double"))


def transform_fundeb(dataframe: DataFrame) -> DataFrame:
    """Soma os meses e transforma os anos selecionados em linhas por município."""

    dataframe = dataframe.groupBy(
        "city_code",
        "city_name",
        "state_abbreviation",
    ).agg(
        *[
            F.coalesce(F.sum(parse_amount(ano)), F.lit(0.0))
            .cast("double")
            .alias(ano)
            for ano in YEARS
        ]
    )
    dataframe = dataframe.select(
        "city_code",
        "city_name",
        "state_abbreviation",
        F.explode(F.array(*[
            F.struct(F.lit(int(ano)).alias("year"), F.col(ano).alias("fundeb_amount"))
            for ano in YEARS
        ])).alias("valor"),
    )
    return dataframe.select(
        F.col("valor.year").alias("year"),
        "city_code",
        "city_name",
        "state_abbreviation",
        F.col("valor.fundeb_amount").alias("value"),
    )


def load(dataframe: DataFrame, bucket_silver: str, folder: str) -> str:
    """Grava o resultado em Parquet na Silver, particionado por ano."""

    path = f"{bucket_silver}/{folder}/fundeb_anual"
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
dataframe = read_fundeb(spark, uri)
dataframe_anual = transform_fundeb(dataframe)
load_path = load(dataframe_anual, BUCKET_SILVER, FOLDER)
log.info(f"Normalizado {uri} -> {load_path}")

job.commit()
