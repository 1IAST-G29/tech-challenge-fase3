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

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'BUCKET_SILVER', 'BUCKET_GOLD'])  # lê os parâmetros passados ao job
JOB_NAME = args['JOB_NAME']  # nome do job, exigido pelo Glue
BUCKET_SILVER = args['BUCKET_SILVER']  # origem: camada Silver no S3
BUCKET_GOLD = args['BUCKET_GOLD']  # destino: camada Gold no S3


# ============================================================
# VARIABLES
# ============================================================

ALUNOS_PATH = "alfabetizacao_inep/aluno"  # base: um registro por aluno e ano
CENSO_PATH = "censo_escolar/censo_escolar_municipal"  # dados municipais, ano anterior ao aluno
FUNDEB_PATH = "fundeb/fundeb_anual"  # dados municipais por nome, ano anterior ao aluno
IDHM_PATH = "idhm/idhm_uf"  # dados estaduais de 2022 a 2024, mesmo ano do aluno
GOLD_PATH = "alfabetizacao/base_alunos"  # destino da base consolidada
METADADOS = ["_source_file", "_ingested_at"]  # colunas técnicas da Silver que não entram na Gold
ACENTUADOS = "ÁÀÂÃÉÊÍÓÔÕÚÇ"  # letras maiúsculas com acento
SEM_ACENTO = "AAAAEEIOOOUC"  # mesmas letras, sem acento, na mesma ordem


# ============================================================
# FUNCTIONS
# ============================================================

def read_silver(spark, bucket_silver: str, path: str) -> DataFrame:
    """Lê uma tabela da Silver em Parquet."""

    return spark.read.parquet(f"{bucket_silver}/{path}")  # lê todas as partições da tabela


def chave_municipio(coluna: str):
    """Padroniza o nome do município em maiúsculas e sem acento, para casar o fundeb com os alunos."""

    return F.translate(F.upper(F.trim(F.col(coluna))), ACENTUADOS, SEM_ACENTO)  # troca acentos por letras simples


def build_gold(alunos: DataFrame, censo: DataFrame, fundeb: DataFrame, idhm: DataFrame) -> DataFrame:
    """Une os datasets à base de alunos, usando o dado do ano anterior, exceto o IDHM, que usa o mesmo ano."""

    base = (alunos
                .drop(*METADADOS)  # remove as colunas técnicas da Silver
                .withColumn("id_municipio", F.col("co_municipio").cast("int"))  # código IBGE do município
                .withColumn("state_code", F.col("co_uf").cast("int"))  # código IBGE da UF
                .withColumn("chave_municipio", chave_municipio("no_municipio"))  # nome padronizado para o fundeb
            )

    # Censo do ano anterior: ao deslocar o ano, o dado de 2022 passa a casar com os alunos de 2023
    censo_ref = censo.withColumn("ano", F.col("ano") + 1)
    base = base.join(F.broadcast(censo_ref), ["ano", "id_municipio"], "left")

    # FUNDEB do ano anterior, casado por UF e nome do município padronizado
    fundeb_ref = (fundeb
                    .withColumn("ano", F.col("year") + 1)  # desloca o ano para o ano do aluno
                    .withColumn("chave_municipio", chave_municipio("city_name"))  # mesmo padrão de nome dos alunos
                    .withColumnRenamed("state_abbreviation", "sg_uf")  # mesma UF usada nos alunos
                    .select("ano", "chave_municipio", "sg_uf", "fundeb_amount")
                )
    base = base.join(F.broadcast(fundeb_ref), ["ano", "chave_municipio", "sg_uf"], "left")

    # IDHM do mesmo ano, por código IBGE da UF
    idhm_ref = idhm.withColumnRenamed("year", "ano")  # o ano do IDHM já é o ano do aluno
    base = base.join(F.broadcast(idhm_ref), ["ano", "state_code"], "left")

    return base.drop("chave_municipio")  # remove a chave auxiliar


def load(dataframe: DataFrame, bucket_gold: str, path: str) -> str:
    """Grava a base consolidada em Parquet na Gold, particionada por ano."""

    destino = f"{bucket_gold}/{path}"  # destino: gold/alfabetizacao/base_alunos
    dataframe.write.mode("overwrite").partitionBy("ano").parquet(destino)  # uma pasta por ano
    return destino


# ============================================================
# EXECUTION
# ============================================================

sc = SparkContext()  # contexto Spark exigido pelo GlueContext
glueContext = GlueContext(sc)  # contexto do Glue
spark = glueContext.spark_session  # sessão Spark para leitura e junção
job = Job(glueContext)  # objeto que controla o ciclo de vida do job
job.init(JOB_NAME, args)  # inicia o job com os parâmetros resolvidos

alunos = read_silver(spark, BUCKET_SILVER, ALUNOS_PATH)  # base de alunos
censo = read_silver(spark, BUCKET_SILVER, CENSO_PATH)  # censo municipal
fundeb = read_silver(spark, BUCKET_SILVER, FUNDEB_PATH)  # fundeb anual por município
idhm = read_silver(spark, BUCKET_SILVER, IDHM_PATH)  # IDHM por UF

gold = build_gold(alunos, censo, fundeb, idhm)  # junta os datasets à base de alunos
load_path = load(gold, BUCKET_GOLD, GOLD_PATH)  # grava na Gold particionado por ano

job.commit()  # finaliza o job e registra o estado no Glue
