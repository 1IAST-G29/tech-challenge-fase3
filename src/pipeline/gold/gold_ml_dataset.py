"""Cria a base Gold destinada ao treinamento de modelos.

O job lê ``alfabetizacao/base_alunos``, produzida por
``gold_consolidacao_alunos.py``, padroniza o alvo para ``alfabetizado`` e
remove identificadores e variáveis que só ficam disponíveis depois da
avaliação.
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
TARGET_CANDIDATES = (
    "student_is_literate",
    "in_alfabetizado",
    "alfabetizado",
)
YEAR_CANDIDATES = ("year", "ano")

# Variáveis que revelam o resultado da avaliação ou foram calculadas a partir dele.
POST_EVALUATION_COLUMNS = {
    "student_exam_score",
    "student_exam_weight",
    "student_is_present_exam",
    "student_took_exam",
    "in_presenca_lp",
    "in_preenchimento_lp",
    "vl_proficiencia_lp",
    "proficiencia",
    "city_percentage_literate_students",
    "city_average_exam_score",
    "state_percentage_literate_students",
    "state_average_exam_score",
}

# Identificadores individuais e nomes de localidades não são atributos adequados
# para o modelo. O código municipal é uma exceção deliberada: o produto será
# usado principalmente nos municípios já conhecidos e o modelo poderá aprender
# o prior histórico de alfabetização de cada município.
IDENTIFIER_COLUMNS = {
    "student_id",
    "school_id",
    "state_code",
    "id_aluno",
    "id_escola",
    "id_municipio",
    "co_municipio",
    "co_uf",
    "city_name",
    "city_name_norm",
    "no_municipio",
}


def read_gold(spark: object, bucket_gold: str, path: str) -> DataFrame:
    """Lê a Gold consolidada em Parquet."""

    return spark.read.parquet(f"{bucket_gold}/{path}")


def find_column(dataframe: DataFrame, candidates: tuple[str, ...], description: str) -> str:
    """Encontra a primeira coluna disponível entre os nomes aceitos."""

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    raise ValueError(
        f"A Gold não possui {description}. "
        f"Colunas aceitas: {', '.join(candidates)}"
    )


def build_ml_dataset(dataframe: DataFrame) -> DataFrame:
    """Converte a Gold consolidada em uma tabela pronta para treinamento."""

    source_target = find_column(dataframe, TARGET_CANDIDATES, "o alvo de alfabetização")
    source_year = find_column(dataframe, YEAR_CANDIDATES, "o ano")

    dataframe = (
        dataframe.withColumn("alfabetizado", F.col(source_target).cast("int"))
        .withColumn("ano", F.col(source_year).cast("int"))
        .filter(F.col("alfabetizado").isin(0, 1))
        .filter(F.col("ano").isNotNull())
    )

    columns_to_drop = (
        POST_EVALUATION_COLUMNS
        | IDENTIFIER_COLUMNS
        | set(TARGET_CANDIDATES)
        | set(YEAR_CANDIDATES)
    )
    columns_to_drop.discard("alfabetizado")
    columns_to_drop.discard("ano")
    existing_columns = [column for column in columns_to_drop if column in dataframe.columns]

    features = dataframe.drop(*existing_columns).select(
        "ano",
        "alfabetizado",
        *[
            column
            for column in dataframe.columns
            if column not in columns_to_drop
            and column not in {"ano", "alfabetizado"}
        ],
    )
    if "city_code" in features.columns:
        features = features.withColumn("city_code", F.col("city_code").cast("string"))
    return features


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
print(
    f"Dataset de treinamento gravado em {load_path}: "
    f"{ml_dataset.count()} registros e {len(ml_dataset.columns)} colunas",
    flush=True,
)

job.commit()
