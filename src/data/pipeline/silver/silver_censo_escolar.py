import logging
import sys
from functools import reduce

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

FOLDER = "censo_escolar"  # pasta da Bronze processada por este job
YEARS = [2022, 2023, 2024]  # anos do recorte, igual ao notebook
CSV_SEPARATOR = ";"  # separador usado nos CSVs da Bronze
CSV_ENCODING = "ISO-8859-1"  # encoding usado nos CSVs da Bronze
COLUNAS_CENSO = [
    "CO_MUNICIPIO",
    "TP_SITUACAO_FUNCIONAMENTO",
    "TP_LOCALIZACAO",
    "QT_MAT_FUND_AI",
    "QT_MAT_FUND_AI_INT",
    "QT_DOC_FUND_AI",
    "QT_TUR_FUND_AI",
    "IN_INTERNET",
    "IN_BIBLIOTECA",
    "IN_SALA_LEITURA",
]  # colunas usadas na base municipal, igual ao notebook


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

def read_censo(spark, uri: str, ano: int) -> DataFrame:
    """Lê o microdados de um ano, mantém as colunas usadas e só escolas em atividade com anos iniciais."""

    dataframe = (spark.read
                    .option("header", True)  # primeira linha são os nomes das colunas
                    .option("sep", CSV_SEPARATOR)  # separador ;
                    .option("encoding", CSV_ENCODING)  # encoding da Bronze
                    .csv(uri)
                )
    dataframe = dataframe.select([F.col(coluna).cast("double").alias(coluna) for coluna in COLUNAS_CENSO])  # números para cálculo
    dataframe = dataframe.filter(
        (F.col("TP_SITUACAO_FUNCIONAMENTO") == 1) & (F.col("QT_MAT_FUND_AI") > 0)  # escolas em atividade com matrículas nos anos iniciais
    )
    return dataframe.withColumn("ano", F.lit(ano))  # guarda o ano de referência


def soma(coluna: str):
    """Soma a coluna por grupo e troca o nulo por zero, como o pandas faz."""

    return F.coalesce(F.sum(coluna), F.lit(0))


def transform_municipal(dataframe: DataFrame) -> DataFrame:
    """Calcula as flags por escola, agrega por município e ano e monta os percentuais e indicadores."""

    dataframe = (dataframe
                    .withColumn("escola_rural", F.when(F.col("TP_LOCALIZACAO") == 2, 1).otherwise(0))  # 1 se a escola é rural
                    .withColumn("espaco_leitura", F.when((F.col("IN_BIBLIOTECA") == 1) | (F.col("IN_SALA_LEITURA") == 1), 1).otherwise(0))  # 1 se tem biblioteca ou sala de leitura
                    .withColumn("matriculas_turmas_zero", F.when(F.col("QT_TUR_FUND_AI") == 0, F.col("QT_MAT_FUND_AI")).otherwise(0))  # matrículas de escolas sem turmas
                    .withColumn("docentes_nulos", F.when(F.col("QT_DOC_FUND_AI").isNull(), 1).otherwise(0))  # 1 se a escola não informou docentes
                )

    municipal = dataframe.groupBy("CO_MUNICIPIO", "ano").agg(
        F.count("*").alias("qtd_escolas"),  # quantidade de escolas
        soma("QT_MAT_FUND_AI").alias("total_matriculas"),  # total de matrículas
        soma("QT_DOC_FUND_AI").alias("total_docentes"),  # total de docentes
        soma("QT_TUR_FUND_AI").alias("total_turmas"),  # total de turmas
        soma("escola_rural").alias("escolas_rurais"),  # escolas rurais
        soma("IN_INTERNET").alias("escolas_internet"),  # escolas com internet
        soma("IN_BIBLIOTECA").alias("escolas_biblioteca"),  # escolas com biblioteca
        soma("IN_SALA_LEITURA").alias("escolas_sala_leitura"),  # escolas com sala de leitura
        soma("espaco_leitura").alias("escolas_espaco_leitura"),  # escolas com biblioteca ou sala de leitura
        soma("QT_MAT_FUND_AI_INT").alias("matriculas_integral"),  # matrículas em tempo integral
        soma("matriculas_turmas_zero").alias("matriculas_turmas_zero"),  # matrículas em escolas sem turmas
        soma("docentes_nulos").alias("qtd_escolas_docentes_nulos"),  # escolas sem docentes informados
    )

    for destino, origem in {
        "pct_escolas_rurais": "escolas_rurais",
        "pct_escolas_internet": "escolas_internet",
        "pct_escolas_biblioteca": "escolas_biblioteca",
        "pct_escolas_sala_leitura": "escolas_sala_leitura",
        "pct_escolas_espaco_leitura": "escolas_espaco_leitura",
    }.items():
        municipal = municipal.withColumn(destino, F.col(origem) / F.col("qtd_escolas") * 100)  # percentual de escolas

    municipal = (municipal
                    .withColumn("pct_matriculas_integral", F.col("matriculas_integral") / F.col("total_matriculas") * 100)  # percentual de matrículas em tempo integral
                    .withColumn("alunos_por_docente", F.when(F.col("total_docentes") != 0, F.col("total_matriculas") / F.col("total_docentes")))  # sem divisão por zero
                    .withColumn("alunos_por_turma", F.when(F.col("total_turmas") != 0, F.col("total_matriculas") / F.col("total_turmas")))  # sem divisão por zero
                    .withColumn("pct_matriculas_escolas_turmas_zero", F.col("matriculas_turmas_zero") / F.col("total_matriculas") * 100)  # percentual de matrículas em escolas sem turmas
                )

    return municipal.select(
        "ano",
        F.col("CO_MUNICIPIO").cast("int").alias("id_municipio"),  # código do município como inteiro
        "qtd_escolas",
        "total_matriculas",
        "total_docentes",
        "total_turmas",
        "pct_escolas_rurais",
        "pct_escolas_internet",
        "pct_escolas_biblioteca",
        "pct_escolas_sala_leitura",
        "pct_escolas_espaco_leitura",
        "pct_matriculas_integral",
        "alunos_por_docente",
        "alunos_por_turma",
        "pct_matriculas_escolas_turmas_zero",
        "qtd_escolas_docentes_nulos",
    )


def load(dataframe: DataFrame, bucket_silver: str, folder: str) -> str:
    """Grava a base municipal em Parquet na Silver, particionada por ano."""

    path = f"{bucket_silver}/{folder}/censo_escolar_municipal"  # destino: silver/censo_escolar/censo_escolar_municipal
    dataframe.write.mode("overwrite").partitionBy("ano").parquet(path)  # uma pasta por ano
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

dataframes = [
    read_censo(spark, f"{BUCKET_BRONZE}/{FOLDER}/{ano}_microdados_ed_basica.csv", ano)  # lê cada ano do recorte
    for ano in YEARS
]
dataframe = reduce(lambda a, b: a.unionByName(b), dataframes)  # empilha os três anos
dataframe_municipal = transform_municipal(dataframe)  # calcula a base municipal por ano
load_path = load(dataframe_municipal, BUCKET_SILVER, FOLDER)  # grava na Silver particionado por ano
log.info(f"Base municipal {YEARS} -> {load_path}")

job.commit()  # finaliza o job e registra o estado no Glue
