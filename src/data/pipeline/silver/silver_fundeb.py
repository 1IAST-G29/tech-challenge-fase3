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

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'BUCKET_BRONZE', 'BUCKET_SILVER'])  # lê os parâmetros passados ao job
JOB_NAME = args['JOB_NAME']  # nome do job, exigido pelo Glue
BUCKET_BRONZE = args['BUCKET_BRONZE']  # origem: camada Bronze no S3
BUCKET_SILVER = args['BUCKET_SILVER']  # destino: camada Silver no S3


# ============================================================
# VARIABLES
# ============================================================

FOLDER = "fundeb"  # pasta da Bronze processada por este job
FILE_NAME = "fundeb-por-municipio.csv"  # arquivo da Bronze usado pelo notebook
CSV_SEPARATOR = ";"  # separador usado no CSV
CSV_ENCODING = "ISO-8859-1"  # encoding usado pelo notebook (latin-1)
YEARS = [str(ano) for ano in range(2023, 2026)]  # anos somados, igual ao notebook


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

def read_fundeb(spark, uri: str) -> DataFrame:
    """Lê o CSV do FUNDEB e mantém só o município, a UF e as colunas dos anos."""

    dataframe = (spark.read
                    .option("header", True)  # primeira linha são os nomes das colunas
                    .option("sep", CSV_SEPARATOR)  # separador ;
                    .option("encoding", CSV_ENCODING)  # encoding latin-1
                    .csv(uri)
                )
    dataframe = dataframe.select(
        F.col("Município").alias("city_name"),  # renomeia o município
        F.col("UF").alias("state_abbreviation"),  # renomeia a UF
        *[F.col(ano) for ano in YEARS],  # mantém só as colunas dos anos
    )
    return dataframe.filter(F.col("city_name").isNotNull())  # remove as linhas vazias do fim do arquivo


def parse_amount(column: str):
    """Converte o texto no formato brasileiro (1.926.974,37) para número; '-' e vazio viram nulo."""

    texto = F.regexp_replace(F.regexp_replace(F.trim(F.col(column)), r"\.", ""), ",", ".")  # tira o ponto de milhar e troca a vírgula por ponto
    return F.when(texto.rlike(r"^-?[0-9]+(\.[0-9]+)?$"), texto.cast("double"))  # converte apenas valores numéricos


def transform_fundeb(dataframe: DataFrame) -> DataFrame:
    """Soma os 12 meses por município e ano e passa as colunas de ano para linhas."""

    dataframe = dataframe.groupBy("city_name", "state_abbreviation").agg(
        *[F.coalesce(F.sum(parse_amount(ano)), F.lit(0.0)).alias(ano) for ano in YEARS]  # sem valor vira 0, como no pandas
    )
    dataframe = dataframe.select(
        "city_name",
        "state_abbreviation",
        F.explode(F.array(*[
            F.struct(F.lit(int(ano)).alias("year"), F.col(ano).alias("fundeb_amount"))  # um par ano/valor por coluna de ano
            for ano in YEARS
        ])).alias("valor"),
    )
    return dataframe.select(
        F.col("valor.year").alias("year"),
        "city_name",
        "state_abbreviation",
        F.col("valor.fundeb_amount").alias("fundeb_amount"),
    )


def load(dataframe: DataFrame, bucket_silver: str, folder: str) -> str:
    """Grava o resultado em Parquet na Silver, particionado por ano."""

    path = f"{bucket_silver}/{folder}/fundeb_anual"  # destino: silver/fundeb/fundeb_anual
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
spark = glueContext.spark_session  # sessão Spark para leitura e transformação
job = Job(glueContext)  # objeto que controla o ciclo de vida do job
job.init(JOB_NAME, args)  # inicia o job com os parâmetros resolvidos

uri = f"{BUCKET_BRONZE}/{FOLDER}/{FILE_NAME}"  # arquivo de origem na Bronze
dataframe = read_fundeb(spark, uri)  # lê e filtra as colunas
dataframe_anual = transform_fundeb(dataframe)  # soma por ano e transforma anos em linhas
load_path = load(dataframe_anual, BUCKET_SILVER, FOLDER)  # grava na Silver particionado por ano
log.info(f"Normalizado {uri} -> {load_path}")

job.commit()  # finaliza o job e registra o estado no Glue
