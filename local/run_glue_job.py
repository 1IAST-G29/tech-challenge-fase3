"""Executa um job Glue localmente usando Spark e stubs mínimos do Glue."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

from pyspark.sql import SparkSession


def install_glue_stubs() -> None:
    """Disponibiliza as classes usadas pelos jobs sem instalar o runtime Glue."""

    glue_module = types.ModuleType("awsglue")
    context_module = types.ModuleType("awsglue.context")
    job_module = types.ModuleType("awsglue.job")
    utils_module = types.ModuleType("awsglue.utils")

    class GlueContext:
        def __init__(self, spark_context: object) -> None:
            self.spark_session = SparkSession.builder.getOrCreate()

    class Job:
        def __init__(self, glue_context: GlueContext) -> None:
            self.glue_context = glue_context

        def init(self, job_name: str, arguments: dict[str, str]) -> None:
            return None

        def commit(self) -> None:
            return None

    def get_resolved_options(arguments: list[str], names: list[str]) -> dict[str, str]:
        values = dict(zip(arguments[1::2], arguments[2::2]))
        result: dict[str, str] = {}
        for name in names:
            option = f"--{name}"
            if option not in values:
                raise ValueError(f"Parâmetro obrigatório ausente: {option}")
            result[name] = values[option]
        return result

    context_module.GlueContext = GlueContext
    job_module.Job = Job
    utils_module.getResolvedOptions = get_resolved_options
    sys.modules["awsglue"] = glue_module
    sys.modules["awsglue.context"] = context_module
    sys.modules["awsglue.job"] = job_module
    sys.modules["awsglue.utils"] = utils_module


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Uso: python local/run_glue_job.py CAMINHO_DO_JOB "
            "--JOB_NAME nome --PARAMETRO valor"
        )

    job_path = Path(sys.argv[1]).resolve()
    if not job_path.is_file():
        raise FileNotFoundError(f"Job não encontrado: {job_path}")

    install_glue_stubs()
    sys.argv = [str(job_path), *sys.argv[2:]]
    runpy.run_path(str(job_path), run_name="__main__")


if __name__ == "__main__":
    main()
