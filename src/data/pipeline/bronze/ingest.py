"""Ingest resources into the Bronze layer using configurable PySpark sources."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Mapping, Protocol
from uuid import uuid4
import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# ============================================================
# VARIABLES
# ============================================================

GCP_PROJECT_DATASET = "basedosdados.br_inep_avaliacao_alfabetizacao"

TABLES = [
    "dicionario",
    "alunos",
    "municipio",
    "uf",
    "meta_alfabetizacao_municipio",
    "meta_alfabetizacao_uf",
    "meta_alfabetizacao_brasil",
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

class ResourceRetriever:
    """Resource Retriever interface."""

    name: str

    def read(self, spark: SparkSession, resource_name: str) -> DataFrame:
        """Return a Spark DataFrame."""


class BigQueryRetriver(ResourceRetriever):
    """Read a table from BigQuery with Spark."""

    name: str = "bigquery"

    def read(self, spark: SparkSession, resource_name: str, options: Mapping[str, str]) -> DataFrame:
        full_resource_path = f"{GCP_PROJECT_DATASET}.{resource_name}"
        reader = spark.read.format("bigquery").option("table", full_resource_path)

        for key, value in options.items():
            reader = reader.option(key, value)
        dataframe = (
            spark.read.format("bigquery")
            .option("table", full_resource_path)
            .option("spark.jars", "/Users/pedrolavor/Workspace/tech-challenge-fase3/src/data/pipeline/bronze/spark-3.5-bigquery-0.45.0.jar")
            .option("parentProject", "pos-tech-ai-scientist")
            .option("credentialsFile", "/Users/pedrolavor/Workspace/tech-challenge-fase3/credentials.json")
            .load()
        )
        return dataframe


class SparkResourceRetriever():
    """Retrieve resource with Spark."""

    sparkSession: SparkSession
    format: str
    default_options: dict[str, str]

    def __init__(self, sparkSession: SparkSession, format: str, default_options: dict[str, str] | None = None) -> None:
        self.sparkSession = sparkSession
        self.format = format
        self.default_options = default_options or {}

    def read(self, options: dict[str, str]) -> DataFrame:
        reader = self.sparkSession.read.format(self.format)

        for key, value in self.default_options.items():
            reader = reader.option(key, value)

        for key, value in options.items():
            reader = reader.option(key, value)
            
        return reader.load()


def create_spark_session(
    runtime: str,
    options: Mapping[str, str] | None = None,
) -> SparkSession:
    """Create Spark for local, Docker, or cluster execution.

    Runtime-specific defaults are kept here so ingestion sources do not need to
    know where Spark is running. Cloud runtimes deliberately do not set a master;
    the cluster or managed service supplies it.
    """

    if runtime not in {"local", "docker", "cloud"}:
        raise SourceConfigurationError(
            f"Runtime desconhecido: {runtime}. Use local, docker ou cloud."
        )

    spark_options = dict(options or {})
    app_name = spark_options.pop("app_name", "bronze-ingestion")
    master = spark_options.pop("master", None)
    if master is None and runtime in {"local", "docker"}:
        master = "local[*]"

    builder = SparkSession.builder.appName(app_name)
    if master is not None:
        builder = builder.master(master)
    for key, value in spark_options.items():
        builder = builder.config(key, value)
    return builder.getOrCreate()



# @dataclass(frozen=True)
# class FilesystemSource:
#     """Read a table from a local CSV or Parquet file with Spark."""

#     spark: SparkSession
#     input_dir: Path
#     name: str = "filesystem"

#     def read(self, table_name: str) -> DataFrame:
#         parquet_path = self.input_dir / f"{table_name}.parquet"
#         csv_path = self.input_dir / f"{table_name}.csv"
#         if parquet_path.exists():
#             return self.spark.read.parquet(str(parquet_path))
#         if csv_path.exists():
#             frame = self.spark.read.option("header", True).option("inferSchema", True).csv(
#                 str(csv_path)
#             )
#             return frame
#         raise SourceNotFoundError(
#             f"Fonte {table_name} nao encontrada em {self.input_dir}. "
#             "Informe um arquivo .parquet ou .csv com o mesmo nome da tabela."
#         )




def create_source(
    source_type: str,
    *,
    spark: SparkSession,
    options: Mapping[str, str] | None = None,
) -> ResourceSource:
    """Build the source selected by configuration or the command line."""

    source_options = options or {}
    if source_type == "filesystem":
        input_dir = source_options.get("path")
        if input_dir is None:
            raise SourceConfigurationError("A fonte filesystem exige a opcao 'path'.")
        return FilesystemSource(spark=spark, input_dir=Path(input_dir))
    if source_type == "bigquery":
        return BigQuerySource(
            spark=spark,
            project=source_options.get("project"),
            dataset=source_options.get("dataset"),
        )
    raise SourceConfigurationError(
        f"Fonte desconhecida: {source_type}. Use filesystem ou bigquery."
    )


def new_run_id() -> str:
    return uuid4().hex


def _checksum(path: str) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    digest = sha256()
    with source.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _add_ingestion_metadata(
    frame: DataFrame,
    source: ResourceSource,
    source_reference: str,
    run_id: str,
) -> DataFrame:
    return (
        frame.withColumn("_source_name", F.lit(source.name))
        .withColumn("_source_path", F.lit(source_reference))
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_run_id", F.lit(run_id))
    )


def _write_partitioned(
    frame: DataFrame,
    output_root: Path,
    table_name: str,
    run_id: str,
) -> None:
    partitions = (
        ["unknown"]
        if "ano" not in frame.columns
        else [row["ano"] for row in frame.select("ano").distinct().collect()]
    )
    for partition in partitions:
        subset = (
            frame.filter(F.col("ano") == partition)
            if partition != "unknown"
            else frame
        )
        target = (
            output_root
            / table_name
            / f"reference_year={partition}"
            / f"run_id={run_id}"
        )
        subset.write.mode("overwrite").parquet(str(target))


def _write_manifest(
    output_path: Path,
    *,
    run_id: str,
    source: ResourceSource,
    source_reference: str,
    row_count: int,
    columns: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "source_name": source.name,
        "source_path": source_reference,
        "source_checksum_sha256": _checksum(source_reference),
        "extracted_at": datetime.now(UTC).isoformat(),
        "row_count": row_count,
        "columns": columns,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest(
    source: ResourceSource,
    output_dir: Path,
    run_id: str,
    tables: tuple[str, ...],
) -> None:
    """Read selected tables from ``source`` and write immutable Bronze files."""

    for table_name in tables:
        frame, source_reference = source.read(table_name)
        bronze = _add_ingestion_metadata(frame, source, source_reference, run_id)
        _write_partitioned(bronze, output_dir / "bronze", table_name, run_id)
        _write_manifest(
            output_dir / "manifests" / run_id / f"{table_name}.json",
            run_id=run_id,
            source=source,
            source_reference=source_reference,
            row_count=frame.count(),
            columns=frame.columns,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingere recursos na camada Bronze com PySpark.")
    parser.add_argument(
        "--runtime",
        choices=("local", "docker", "cloud"),
        default="local",
        help="Ambiente de execucao do Spark.",
    )
    parser.add_argument("--source", choices=("filesystem", "bigquery"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--spark-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Configuracao Spark generica, repetivel e independente do ambiente.",
    )
    parser.add_argument(
        "--source-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Opcao da fonte, repetivel e independente da implementacao.",
    )
    parser.add_argument("--run-id", default=new_run_id())
    parser.add_argument(
        "--tables",
        default=",".join(TABLES),
        help="Lista de tabelas separada por virgula.",
    )
    return parser.parse_args()


def parse_source_options(values: list[str]) -> dict[str, str]:
    """Convert generic KEY=VALUE options into a source configuration mapping."""

    options: dict[str, str] = {}
    for value in values:
        key, separator, option_value = value.partition("=")
        if not separator or not key or not option_value:
            raise Exception(
                f"Opcao de fonte invalida: {value}. Use o formato KEY=VALUE."
            )
        options[key] = option_value
    return options


def main() -> None:
    arguments = parse_args()
    source_options = parse_source_options(arguments.source_option)
    spark_options = parse_source_options(arguments.spark_option)
    spark = create_spark_session(arguments.runtime, spark_options)
    try:
        source = create_source(
            arguments.source,
            spark=spark,
            options=source_options,
        )
        selected_tables = tuple(
            table.strip() for table in arguments.tables.split(",") if table.strip()
        )
        ingest(source, arguments.output_dir, arguments.run_id, selected_tables)
    finally:
        spark.stop()


if __name__ == "__main__":
    import basedosdados as bd

    spark = (
        SparkSession.builder
        .appName("bronze-ingestion")
        .config("spark.jars", "/Users/pedrolavor/Workspace/tech-challenge-fase3/src/data/pipeline/bronze/spark-3.5-bigquery-0.45.0.jar")
        # Força o Spark a escutar apenas na máquina local (evita problemas de rede)
        .config("spark.driver.host", "localhost") \
        .master("local[*]")
        .getOrCreate()
    )

    # dataframe = bd.read_table(
    #     dataset_id="br_inep_avaliacao_alfabetizacao",
    #     table_id="uf",
    #     billing_project_id="pos-tech-ai-scientist",
    #     query_project_id="basedosdados",
    # )
    # arguments = parse_args()

    # reader = spark.read.format("bigquery")
    # for key, value in arguments.spark_option.items():
    #     reader = reader.option(key, value)

    # full_resource_path = f"{GCP_PROJECT_DATASET}.uf"
    # dataframe = reader.option("table", full_resource_path).load()

    arguments = parse_args()
    source_options = parse_source_options(arguments.source_option)
    resource_retriever = SparkResourceRetriever(
        sparkSession=spark,
        format="bigquery",
        default_options=source_options
    )
    dataframe = resource_retriever.read({
        "table": f"{GCP_PROJECT_DATASET}.uf"
    })

    print(dataframe)
    dataframe.show(5)
