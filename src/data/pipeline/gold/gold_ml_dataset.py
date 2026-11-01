"""Cria a base Gold destinada ao treinamento do modelo.

O job lê a Gold consolidada por aluno, padroniza o alvo para ``alfabetizado``
e remove variáveis que causariam vazamento de dados ou que identificam
diretamente uma pessoa. A saída ainda preserva chaves territoriais úteis para
auditoria e agregações posteriores.
"""

import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


args = getResolvedOptions(sys.argv, ["JOB_NAME", "BUCKET_GOLD"])
JOB_NAME = args["JOB_NAME"]
BUCKET_GOLD = args["BUCKET_GOLD"]

INPUT_PATH = "alfabetizacao/base_alunos"
OUTPUT_PATH = "ml/ml_dataset_alunos"
TARGET_CANDIDATES = ("in_alfabetizado", "alfabetizado")

# Estas colunas revelam o resultado da avaliação ou só existem depois dela.
PROHIBITED_COLUMNS = {
    "vl_proficiencia_lp",
    "proficiencia",
    "co_caderno_lp",
    "in_presenca_lp",
    "in_preenchimento_lp",
}

# IDs servem para auditoria e joins, mas não devem ser usados diretamente como atributos.
AUDIT_COLUMNS = {
    "ano",
    "id_aluno",
    "id_escola",
    "id_municipio",
    "state_code",
    "sg_uf",
    "no_municipio",
}


def read_gold(spark: object, bucket_gold: str, path: str) -> DataFrame:
    """Lê a base Gold consolidada em Parquet."""

    return spark.read.parquet(f"{bucket_gold}/{path}")


def find_target(dataframe: DataFrame) -> str:
    """Encontra o nome usado para o alvo na Gold consolidada."""

    for candidate in TARGET_CANDIDATES:
        if candidate in dataframe.columns:
            return candidate
    raise ValueError(
        "A Gold não possui o alvo esperado. "
        f"Colunas aceitas: {', '.join(TARGET_CANDIDATES)}"
    )


def build_ml_dataset(dataframe: DataFrame) -> DataFrame:
    """Seleciona o alvo, as variáveis permitidas e as chaves de auditoria."""

    source_target = find_target(dataframe)
    dataframe = dataframe.withColumn(
        "alfabetizado",
        F.col(source_target).cast("int"),
    )
    dataframe = dataframe.filter(F.col("alfabetizado").isin(0, 1))

    columns_to_drop = PROHIBITED_COLUMNS | set(TARGET_CANDIDATES)
    columns_to_drop.discard(source_target)
    columns_to_drop = {
        column for column in columns_to_drop if column in dataframe.columns
    }
    dataframe = dataframe.drop(*columns_to_drop)

    # Remove identificadores individuais; chaves territoriais ficam para auditoria.
    individual_ids = {
        "id_aluno",
        "id_escola",
        "co_municipio",
        "co_uf",
    }
    existing_ids = [column for column in individual_ids if column in dataframe.columns]
    return dataframe.drop(*existing_ids)


def load(dataframe: DataFrame, bucket_gold: str, path: str) -> str:
    """Grava o dataset de treinamento em Parquet, particionado por ano."""

    destination = f"{bucket_gold}/{path}"
    dataframe.write.mode("overwrite").partitionBy("ano").parquet(destination)
    return destination


sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(JOB_NAME, args)

gold = read_gold(spark, BUCKET_GOLD, INPUT_PATH)
ml_dataset = build_ml_dataset(gold)
load_path = load(ml_dataset, BUCKET_GOLD, OUTPUT_PATH)
print(f"Dataset de treinamento gravado em {load_path}", flush=True)

job.commit()
