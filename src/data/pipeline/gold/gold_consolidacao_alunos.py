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
#   --BUCKET_SILVER s3://bucket/silver
#   --BUCKET_GOLD   s3://bucket/gold

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'BUCKET_SILVER', 'BUCKET_GOLD'])
JOB_NAME = args['JOB_NAME']
BUCKET_SILVER = args['BUCKET_SILVER']
BUCKET_GOLD = args['BUCKET_GOLD']


# ============================================================
# VARIABLES
# ============================================================

ALUNOS_PATH = "alfabetizacao_inep/aluno"
CENSO_PATH = "censo_escolar/censo_escolar_municipal"
CENSO_2022_PATH = "censo_2022/censo_2022.csv"
FUNDEB_PATH = "fundeb/fundeb_anual"
IDHM_PATH = "idhm/idhm_uf"
GOLD_PATH = "alfabetizacao/base_alunos"
METADADOS = ["_source_file", "_ingested_at"]
ACENTUADOS = "ÁÀÂÃÉÊÍÓÔÕÚÇ"
SEM_ACENTO = "AAAAEEIOOOUC"


# ============================================================
# FUNCTIONS
# ============================================================

def read_silver(spark, bucket_silver: str, path: str) -> DataFrame:
    """Lê uma tabela da Silver em Parquet."""

    return spark.read.parquet(f"{bucket_silver}/{path}")


def read_silver_csv(spark, bucket_silver: str, path: str) -> DataFrame:
    """Lê uma tabela CSV da camada Silver."""

    return (
        spark.read
        .option("header", True)
        .option("sep", ";")
        .option("encoding", "UTF-8")
        .csv(f"{bucket_silver}/{path}")
    )


def chave_municipio(coluna: str):
    """Padroniza o nome do município em maiúsculas e sem acento, para casar o fundeb com os alunos."""

    return F.translate(F.upper(F.trim(F.col(coluna))), ACENTUADOS, SEM_ACENTO)


def build_gold(
    alunos: DataFrame,
    censo: DataFrame,
    censo_2022: DataFrame,
    fundeb: DataFrame,
    idhm: DataFrame,
) -> DataFrame:
    """Une os datasets à base de alunos, usando o dado do ano anterior, exceto o IDHM, que usa o mesmo ano."""

    base = (alunos
                .drop(*METADADOS)
                .withColumn("id_municipio", F.col("co_municipio").cast("int"))
                .withColumn("state_code", F.col("co_uf").cast("int"))
                .withColumn("chave_municipio", chave_municipio("no_municipio"))
            )

    # Censo do ano anterior: ao deslocar o ano, o dado de 2022 passa a casar com os alunos de 2023
    censo_ref = censo.withColumn("ano", F.col("ano") + 1)
    base = base.join(F.broadcast(censo_ref), ["ano", "id_municipio"], "left")

    # Dados raciais do Censo 2022: indicadores municipais associados pelo código IBGE.
    censo_2022_ref = censo_2022.withColumn(
        "id_municipio",
        F.col("city_code").cast("int"),
    ).drop("city_code")
    base = base.join(F.broadcast(censo_2022_ref), ["id_municipio"], "left")

    # FUNDEB do ano anterior, casado por UF e nome do município padronizado
    fundeb_ref = (fundeb
                    .withColumn("ano", F.col("year") + 1)
                    .withColumn("chave_municipio", chave_municipio("city_name"))
                    .withColumnRenamed("state_abbreviation", "sg_uf")
                    .select(
                        "ano",
                        "chave_municipio",
                        "sg_uf",
                        F.col("value").alias("fundeb_amount"),
                    )
                )
    base = base.join(F.broadcast(fundeb_ref), ["ano", "chave_municipio", "sg_uf"], "left")

    # IDHM do mesmo ano, por código IBGE da UF
    idhm_ref = idhm.withColumnRenamed("year", "ano")
    base = base.join(F.broadcast(idhm_ref), ["ano", "state_code"], "left")

    return base.drop("chave_municipio")


def load(dataframe: DataFrame, bucket_gold: str, path: str) -> str:
    """Grava a base consolidada em Parquet na Gold, particionada por ano."""

    destino = f"{bucket_gold}/{path}"
    dataframe.write.mode("overwrite").partitionBy("ano").parquet(destino)
    return destino


# ============================================================
# EXECUTION
# ============================================================

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(JOB_NAME, args)

alunos = read_silver(spark, BUCKET_SILVER, ALUNOS_PATH)
censo = read_silver(spark, BUCKET_SILVER, CENSO_PATH)
censo_2022 = read_silver_csv(spark, BUCKET_SILVER, CENSO_2022_PATH)
fundeb = read_silver(spark, BUCKET_SILVER, FUNDEB_PATH)
idhm = read_silver(spark, BUCKET_SILVER, IDHM_PATH)

gold = build_gold(alunos, censo, censo_2022, fundeb, idhm)
load_path = load(gold, BUCKET_GOLD, GOLD_PATH)

job.commit()
