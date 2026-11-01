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

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'BUCKET_BRONZE', 'BUCKET_SILVER'])
JOB_NAME = args['JOB_NAME']
BUCKET_BRONZE = args['BUCKET_BRONZE']
BUCKET_SILVER = args['BUCKET_SILVER']


# ============================================================
# VARIABLES
# ============================================================

FOLDER = "censo_escolar"
YEARS = [2022, 2023, 2024]
YEAR_2025 = 2025
CSV_SEPARATOR = ";"
CSV_ENCODING = "ISO-8859-1"
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
]


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

def read_censo(spark, uri: str, ano: int) -> DataFrame:
    """Lê o microdados de um ano, mantém as colunas usadas e só escolas em atividade com anos iniciais."""

    dataframe = (spark.read
                    .option("header", True)
                    .option("sep", CSV_SEPARATOR)
                    .option("encoding", CSV_ENCODING)
                    .csv(uri)
                )
    dataframe = dataframe.select([F.col(coluna).cast("double").alias(coluna) for coluna in COLUNAS_CENSO])
    dataframe = dataframe.filter(
        (F.col("TP_SITUACAO_FUNCIONAMENTO") == 1) & (F.col("QT_MAT_FUND_AI") > 0)
    )
    return dataframe.withColumn("ano", F.lit(ano))


def read_censo_2025(spark, folder_uri: str) -> DataFrame:
    """Lê os três arquivos de 2025 e os consolida no formato dos microdados anteriores."""

    def read_csv(file_name: str, columns: list[str]) -> DataFrame:
        return (
            spark.read
            .option("header", True)
            .option("sep", CSV_SEPARATOR)
            .option("encoding", CSV_ENCODING)
            .csv(f"{folder_uri}/{file_name}")
            .select([
                F.col(column).cast("double").alias(column)
                for column in columns
            ])
        )

    school = read_csv(
        "Tabela_Escola_2025_V2.csv",
        [
            "CO_ENTIDADE",
            "CO_MUNICIPIO",
            "TP_SITUACAO_FUNCIONAMENTO",
            "TP_LOCALIZACAO",
            "IN_INTERNET",
            "IN_BIBLIOTECA",
            "IN_SALA_LEITURA",
            "QT_PROF_PEDAGOGIA",
        ],
    )
    enrollment = read_csv(
        "Tabela_Matricula_2025_V2.csv",
        ["CO_ENTIDADE", "QT_MAT_FUND_AI", "QT_MAT_FUND_AI_INT"],
    ).groupBy("CO_ENTIDADE").agg(
        F.sum("QT_MAT_FUND_AI").alias("QT_MAT_FUND_AI"),
        F.sum("QT_MAT_FUND_AI_INT").alias("QT_MAT_FUND_AI_INT"),
    )
    classes = read_csv(
        "Tabela_Turma_2025_V2.csv",
        ["CO_ENTIDADE", "QT_TUR_FUND_AI"],
    ).groupBy("CO_ENTIDADE").agg(
        F.sum("QT_TUR_FUND_AI").alias("QT_TUR_FUND_AI"),
    )

    return (
        school.join(enrollment, "CO_ENTIDADE", "left")
        .join(classes, "CO_ENTIDADE", "left")
        .withColumn("QT_MAT_FUND_AI", F.coalesce(F.col("QT_MAT_FUND_AI"), F.lit(0)))
        .withColumn("QT_MAT_FUND_AI_INT", F.coalesce(F.col("QT_MAT_FUND_AI_INT"), F.lit(0)))
        .withColumn("QT_TUR_FUND_AI", F.coalesce(F.col("QT_TUR_FUND_AI"), F.lit(0)))
        .withColumn("QT_DOC_FUND_AI", F.coalesce(F.col("QT_PROF_PEDAGOGIA"), F.lit(0)))
        .filter(
            (F.col("TP_SITUACAO_FUNCIONAMENTO") == 1)
            & (F.col("QT_MAT_FUND_AI") > 0)
        )
        .withColumn("ano", F.lit(YEAR_2025))
        .select([F.col(column) for column in COLUNAS_CENSO] + [F.col("ano")])
    )


def soma(coluna: str):
    """Soma a coluna por grupo e troca o nulo por zero, como o pandas faz."""

    return F.coalesce(F.sum(coluna), F.lit(0))


def transform_municipal(dataframe: DataFrame) -> DataFrame:
    """Calcula as flags por escola, agrega por município e ano e monta os percentuais e indicadores."""

    dataframe = (dataframe
                    .withColumn("escola_rural", F.when(F.col("TP_LOCALIZACAO") == 2, 1).otherwise(0))
                    .withColumn("espaco_leitura", F.when((F.col("IN_BIBLIOTECA") == 1) | (F.col("IN_SALA_LEITURA") == 1), 1).otherwise(0))
                    .withColumn("matriculas_turmas_zero", F.when(F.col("QT_TUR_FUND_AI") == 0, F.col("QT_MAT_FUND_AI")).otherwise(0))
                    .withColumn("docentes_nulos", F.when(F.col("QT_DOC_FUND_AI").isNull(), 1).otherwise(0))
                )

    municipal = dataframe.groupBy("CO_MUNICIPIO", "ano").agg(
        F.count("*").alias("qtd_escolas"),
        soma("QT_MAT_FUND_AI").alias("total_matriculas"),
        soma("QT_DOC_FUND_AI").alias("total_docentes"),
        soma("QT_TUR_FUND_AI").alias("total_turmas"),
        soma("escola_rural").alias("escolas_rurais"),
        soma("IN_INTERNET").alias("escolas_internet"),
        soma("IN_BIBLIOTECA").alias("escolas_biblioteca"),
        soma("IN_SALA_LEITURA").alias("escolas_sala_leitura"),
        soma("espaco_leitura").alias("escolas_espaco_leitura"),
        soma("QT_MAT_FUND_AI_INT").alias("matriculas_integral"),
        soma("matriculas_turmas_zero").alias("matriculas_turmas_zero"),
        soma("docentes_nulos").alias("qtd_escolas_docentes_nulos"),
    )

    for destino, origem in {
        "pct_escolas_rurais": "escolas_rurais",
        "pct_escolas_internet": "escolas_internet",
        "pct_escolas_biblioteca": "escolas_biblioteca",
        "pct_escolas_sala_leitura": "escolas_sala_leitura",
        "pct_escolas_espaco_leitura": "escolas_espaco_leitura",
    }.items():
        municipal = municipal.withColumn(destino, F.col(origem) / F.col("qtd_escolas") * 100)

    municipal = (municipal
                    .withColumn("pct_matriculas_integral", F.col("matriculas_integral") / F.col("total_matriculas") * 100)
                    .withColumn("alunos_por_docente", F.when(F.col("total_docentes") != 0, F.col("total_matriculas") / F.col("total_docentes")))
                    .withColumn("alunos_por_turma", F.when(F.col("total_turmas") != 0, F.col("total_matriculas") / F.col("total_turmas")))
                    .withColumn("pct_matriculas_escolas_turmas_zero", F.col("matriculas_turmas_zero") / F.col("total_matriculas") * 100)
                )

    return municipal.select(
        "ano",
        F.col("CO_MUNICIPIO").cast("int").alias("id_municipio"),
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

    path = f"{bucket_silver}/{folder}/censo_escolar_municipal"
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

dataframes = [
    read_censo(spark, f"{BUCKET_BRONZE}/{FOLDER}/{ano}_microdados_ed_basica.csv", ano)
    for ano in YEARS
]
dataframes.append(read_censo_2025(spark, f"{BUCKET_BRONZE}/{FOLDER}"))
dataframe = reduce(lambda a, b: a.unionByName(b), dataframes)
dataframe_municipal = transform_municipal(dataframe)
load_path = load(dataframe_municipal, BUCKET_SILVER, FOLDER)
log.info(f"Base municipal {YEARS} -> {load_path}")

job.commit()
