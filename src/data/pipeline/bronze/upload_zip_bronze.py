import os
import shutil
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

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'ZIP_PATH', 'BUCKET_BRONZE'])
JOB_NAME = args['JOB_NAME']
ZIP_PATH = args['ZIP_PATH']
BUCKET_BRONZE = args['BUCKET_BRONZE']


def create_s3_client() -> object:
    """Cria um cliente S3, aceitando um endpoint MinIO em testes locais."""

    endpoint = os.getenv("LOCAL_S3_ENDPOINT")
    if endpoint:
        return boto3.client("s3", endpoint_url=endpoint)
    return boto3.client("s3")


# ============================================================
# FUNCTIONS
# ============================================================

def download_from_s3(s3_uri: str, destination: Path) -> None:
    """Baixa o ZIP do S3 para o disco local do worker."""

    parsed = urlparse(s3_uri)
    if parsed.scheme == "file":
        shutil.copyfile(parsed.path, destination)
        return

    s3 = create_s3_client()
    s3.download_file(parsed.netloc, parsed.path.lstrip("/"), str(destination))


def upload_zip(zip_path: Path, bucket_uri: str) -> None:
    """Lê cada arquivo do ZIP e envia para o S3, preservando a estrutura de pastas."""

    parsed = urlparse(bucket_uri)
    prefix = Path(parsed.path.strip("/"))
    if parsed.scheme == "file":
        with ZipFile(zip_path) as zip_file:
            for member in zip_file.infolist():
                if member.is_dir() or member.filename.startswith("__MACOSX/"):
                    continue
                destination = prefix / member.filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
        return

    s3 = create_s3_client()
    with ZipFile(zip_path) as zip_file:
        for member in zip_file.infolist():
            if member.is_dir() or member.filename.startswith("__MACOSX/"):
                continue
            key = (prefix / member.filename).as_posix()
            with zip_file.open(member) as arquivo:
                s3.upload_fileobj(arquivo, parsed.netloc, key)
            print(f"Enviado s3://{parsed.netloc}/{key}", flush=True)


# ============================================================
# EXECUTION
# ============================================================

sc = SparkContext()
glueContext = GlueContext(sc)
job = Job(glueContext)
job.init(JOB_NAME, args)

with tempfile.TemporaryDirectory() as temp_dir:
    zip_path = Path(temp_dir) / Path(urlparse(ZIP_PATH).path).name
    print(f"Baixando {ZIP_PATH}...", flush=True)
    download_from_s3(ZIP_PATH, zip_path)
    upload_zip(zip_path, BUCKET_BRONZE)

job.commit()
