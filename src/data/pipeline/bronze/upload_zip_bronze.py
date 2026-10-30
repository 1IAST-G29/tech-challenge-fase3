import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile

import boto3
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext

# ============================================================
# JOB PARAMETERS
# ============================================================
#
#   --ZIP_PATH      s3://bucket/origem/arquivo.zip
#   --BUCKET_BRONZE s3://bucket/bronze

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'ZIP_PATH', 'BUCKET_BRONZE'])  # lê os parâmetros passados ao job
JOB_NAME = args['JOB_NAME']  # nome do job, exigido pelo Glue
ZIP_PATH = args['ZIP_PATH']  # URI do ZIP de origem no S3
BUCKET_BRONZE = args['BUCKET_BRONZE']  # destino da camada Bronze no S3


# ============================================================
# FUNCTIONS
# ============================================================

def download_from_s3(s3_uri: str, destination: Path) -> None:
    """Baixa o ZIP do S3 para o disco local do worker."""

    parsed = urlparse(s3_uri)  # separa o bucket e a chave da URI s3://
    s3 = boto3.client("s3")  # cliente S3 usando a role do job do Glue
    s3.download_file(parsed.netloc, parsed.path.lstrip("/"), str(destination))  # baixa o arquivo para o disco local


def upload_zip(zip_path: Path, bucket_uri: str) -> None:
    """Lê cada arquivo do ZIP e envia para o S3, preservando a estrutura de pastas."""

    parsed = urlparse(bucket_uri)  # separa o nome do bucket e o prefixo da URI s3://
    prefix = Path(parsed.path.strip("/"))  # pasta dentro do bucket
    s3 = boto3.client("s3")  # cliente S3 usando a role do job do Glue
    with ZipFile(zip_path) as zip_file:  # abre o ZIP sem extraí-lo em disco
        for member in zip_file.infolist():  # percorre cada entrada do ZIP
            if member.is_dir() or member.filename.startswith("__MACOSX/"):  # ignora pastas e metadados do macOS
                continue  # pula para a próxima entrada
            key = (prefix / member.filename).as_posix()  # chave S3 no formato POSIX
            with zip_file.open(member) as arquivo:  # lê o arquivo do ZIP em blocos
                s3.upload_fileobj(arquivo, parsed.netloc, key)  # envia o arquivo para o S3
            print(f"Enviado s3://{parsed.netloc}/{key}", flush=True)  # registra o envio no log do Glue


# ============================================================
# EXECUTION
# ============================================================

sc = SparkContext()  # contexto Spark exigido pelo GlueContext
glueContext = GlueContext(sc)  # contexto do Glue
job = Job(glueContext)  # objeto que controla o ciclo de vida do job
job.init(JOB_NAME, args)  # inicia o job com os parâmetros resolvidos

with tempfile.TemporaryDirectory() as temp_dir:  # diretório temporário local, removido ao final
    zip_path = Path(temp_dir) / Path(urlparse(ZIP_PATH).path).name  # caminho local do ZIP
    print(f"Baixando {ZIP_PATH}...", flush=True)
    download_from_s3(ZIP_PATH, zip_path)  # baixa o ZIP do S3
    upload_zip(zip_path, BUCKET_BRONZE)  # descomprime e envia os arquivos para a Bronze

job.commit()  # finaliza o job e registra o estado no Glue
