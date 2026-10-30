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

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'BUCKET_BRONZE', 'BUCKET_SILVER'])  # lê os parâmetros passados ao job
JOB_NAME = args['JOB_NAME']  # nome do job, exigido pelo Glue
BUCKET_BRONZE = args['BUCKET_BRONZE']  # origem: camada Bronze no S3
BUCKET_SILVER = args['BUCKET_SILVER']  # destino: camada Silver no S3


# ============================================================
# VARIABLES
# ============================================================

FOLDER = "alfabetizacao_inep"  # pasta da Bronze processada por este job
CSV_SEPARATOR = ";"  # separador usado nos CSVs da Bronze
CSV_ENCODING = "ISO-8859-1"  # cobre os arquivos ASCII e ISO-8859 da Bronze


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

def list_files(bucket_uri: str, folder: str) -> list[str]:
    """Lista as URIs de todos os arquivos de uma pasta da Bronze."""

    parsed = urlparse(bucket_uri)  # separa o bucket e o prefixo da URI
    prefix = f"{parsed.path.strip('/')}/{folder}/".lstrip("/")  # prefixo da pasta dentro do bucket
    s3 = boto3.client("s3")  # cliente S3 usando a role do job do Glue
    paginator = s3.get_paginator("list_objects_v2")  # pagina a listagem para pastas com muitos arquivos
    uris = []  # acumula as URIs encontradas
    for page in paginator.paginate(Bucket=parsed.netloc, Prefix=prefix):  # percorre todas as páginas
        for item in page.get("Contents", []):  # cada objeto da pasta
            uris.append(f"s3://{parsed.netloc}/{item['Key']}")  # monta a URI completa
    return uris


def normalize_column(name: str) -> str:
    """Converte o nome de uma coluna para snake_case, sem acentos ou caracteres especiais."""

    sem_acento = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")  # remove acentos
    return re.sub(r"[^a-z0-9]+", "_", sem_acento.lower()).strip("_")  # troca símbolos por "_" e tira as bordas


def normalize(dataframe: DataFrame, source_file: str) -> DataFrame:
    """Padroniza nomes de colunas, remove espaços extras e valores vazios e elimina duplicados."""

    dataframe = dataframe.toDF(*[normalize_column(column) for column in dataframe.columns])  # padroniza os nomes das colunas
    dataframe = dataframe.select([
        F.when(F.trim(F.col(column)) == "", None).otherwise(F.trim(F.col(column))).alias(column)  # trim e vazio vira nulo
        for column in dataframe.columns
    ])
    dataframe = dataframe.dropDuplicates()  # remove linhas repetidas
    dataframe = (dataframe
                    .withColumn("_source_file", F.lit(source_file))  # guarda o arquivo de origem
                    .withColumn("_ingested_at", F.current_timestamp())  # guarda o momento do processamento
                )
    return dataframe


def load(dataframe: DataFrame, bucket_silver: str, folder: str, entity: str) -> str:
    """Grava a entidade concatenada em Parquet na Silver, particionada por ano."""

    path = f"{bucket_silver}/{folder}/{entity}"  # destino: silver/pasta/entidade
    dataframe.write.mode("overwrite").partitionBy("ano").parquet(path)  # uma pasta por ano dentro da entidade
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
spark = glueContext.spark_session  # sessão Spark para leitura dos arquivos
job = Job(glueContext)  # objeto que controla o ciclo de vida do job
job.init(JOB_NAME, args)  # inicia o job com os parâmetros resolvidos

entities = {}  # agrupa os DataFrames por entidade, ex: aluno, estado, municipio

for uri in list_files(BUCKET_BRONZE, FOLDER):  # processa cada arquivo da pasta
    file_name = Path(uri).name  # nome do arquivo, ex: 2025_aluno.csv
    if not file_name.lower().endswith(".csv"):  # ignora arquivos que não são CSV
        log.warning(f"Arquivo ignorado (formato não suportado): {uri}")
        continue  # pula para o próximo arquivo
    ano, entity = Path(file_name).stem.split("_", 1)  # separa o ano e a entidade do nome, ex: 2025 e aluno
    dataframe = (spark.read
                    .option("header", True)  # primeira linha são os nomes das colunas
                    .option("sep", CSV_SEPARATOR)  # separador ;
                    .option("encoding", CSV_ENCODING)  # encoding dos arquivos
                    .csv(uri)
                )
    dataframe_normalized = normalize(dataframe, uri).withColumn("ano", F.lit(int(ano)))  # padroniza e adiciona o ano para particionar
    entities.setdefault(entity, []).append(dataframe_normalized)  # guarda para concatenar com os outros anos

for entity, dataframes in entities.items():  # uma saída por entidade
    dataframe = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dataframes)  # concatena todos os anos, preenchendo colunas ausentes com nulo
    load_path = load(dataframe, BUCKET_SILVER, FOLDER, entity)  # grava na Silver particionado por ano
    log.info(f"Entidade {entity}: {len(dataframes)} arquivos -> {load_path}")

job.commit()  # finaliza o job e registra o estado no Glue
