"""Treina um classificador Spark ML com transformações para variáveis assimétricas.

Uso local:
    spark-submit src/modeling/train_model_spark_transformed.py \
        --input local-data/gold/ml/ml_dataset_alunos \
        --output artifacts/model-spark

Uso no AWS Glue:
    --JOB_NAME <nome> \
    --BUCKET_GOLD s3://<bucket>/gold \
    --BUCKET_MODEL s3://<bucket>/modeling

O job usa o último ano como teste temporal, compara classificadores distribuídos
e não converte a Gold para Pandas no driver. Antes do treinamento, aplica
one-hot encoding em ``education_network_type`` e ``log1p`` nas variáveis
assimétricas definidas em ``LOG1P_COLUMNS``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType

TARGET = "alfabetizado"
YEAR = "ano"
GLUE_INPUT_PATH = "ml/ml_dataset_alunos"
GLUE_OUTPUT_PATH = "alfabetizacao-spark-transformed/runs"
FORBIDDEN_COLUMNS = {
    YEAR,
    "proficiencia",
    "vl_proficiencia_lp",
    "id_aluno",
    "id_escola",
    "id_municipio",
    "state_code",
    "co_municipio",
    "co_uf",
}
GROUP_COLUMNS = ("city_code", "state_abbreviation", "education_network_type")
FORCED_CATEGORICAL_COLUMNS = {"education_network_type"}
LOG1P_COLUMNS = {
    "households",
    "area",
    "population_total",
    "fundeb_amount",
    "school_count",
    "total_enrollments",
    "total_teachers",
    "total_classes",
    "pct_rural_schools",
    "pct_full_time_enrollments",
    "pct_enrollments_zero_class_schools",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Diretório Parquet da ml_dataset_alunos.")
    parser.add_argument("--output", required=True, help="Diretório de saída dos artefatos.")
    parser.add_argument("--num-trees", type=int, default=200, help="Número de árvores da floresta.")
    parser.add_argument("--max-depth", type=int, default=12, help="Profundidade máxima de cada árvore.")
    parser.add_argument(
        "--gbt-max-iter",
        type=int,
        default=200,
        help="Número máximo de iterações das árvores de gradiente.",
    )
    return parser.parse_args()


def normalize_uri(uri: str) -> str:
    """Converte URI local para um caminho aceito pelo Spark."""

    return uri.removeprefix("file://")


def split_temporally(dataframe: DataFrame) -> tuple[DataFrame, DataFrame, list[int], int]:
    """Reserva o último ano para teste, sem misturar dados futuros no treino."""

    if YEAR not in dataframe.columns:
        raise ValueError(f"A base precisa conter a coluna {YEAR!r}.")

    years = [
        row[YEAR]
        for row in dataframe.select(F.col(YEAR).cast("int").alias(YEAR)).where(F.col(YEAR).isNotNull()).distinct().collect()
    ]
    years = sorted(years)
    if len(years) < 2:
        raise ValueError("São necessários pelo menos dois anos para a validação temporal.")

    test_year = years[-1]
    return (
        dataframe.where(F.col(YEAR).cast("int") < test_year),
        dataframe.where(F.col(YEAR).cast("int") == test_year),
        years[:-1],
        test_year,
    )


def prepare_dataset(dataframe: DataFrame) -> tuple[DataFrame, list[str], list[str]]:
    """Aplica log1p e separa os atributos numéricos dos categóricos."""

    if TARGET not in dataframe.columns:
        raise ValueError(f"A base precisa conter a coluna alvo {TARGET!r}.")

    dataframe = (
        dataframe.withColumn("label", F.col(TARGET).cast("double"))
        .where(F.col("label").isin(0.0, 1.0))
    )
    columns_to_drop = FORBIDDEN_COLUMNS | {TARGET, "label"}
    feature_columns = [column for column in dataframe.columns if column not in columns_to_drop]
    if not feature_columns:
        raise ValueError("Nenhuma feature disponível para o treinamento.")
    missing_log_columns = sorted(LOG1P_COLUMNS - set(feature_columns))
    if missing_log_columns:
        raise ValueError(
            "A base não possui as colunas necessárias para a transformação logarítmica: "
            + ", ".join(missing_log_columns)
        )
    missing_categorical_columns = sorted(FORCED_CATEGORICAL_COLUMNS - set(feature_columns))
    if missing_categorical_columns:
        raise ValueError(
            "A base não possui as colunas categóricas obrigatórias: "
            + ", ".join(missing_categorical_columns)
        )
    for column in sorted(LOG1P_COLUMNS):
        dataframe = dataframe.withColumn(column, F.log1p(F.col(column)))

    type_by_column = dict(dataframe.dtypes)
    numeric_columns = [
        field.name
        for field in dataframe.schema.fields
        if (
            field.name in feature_columns
            and field.name not in FORCED_CATEGORICAL_COLUMNS
            and isinstance(field.dataType, NumericType)
        )
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    # StringIndexer trabalha de forma consistente quando campos categóricos possuem
    # representação textual, inclusive para códigos numéricos recebidos como string.
    for column in categorical_columns:
        if type_by_column[column] != "string":
            dataframe = dataframe.withColumn(column, F.col(column).cast(StringType()))

    return dataframe.select("label", YEAR, *feature_columns), numeric_columns, categorical_columns


def add_class_weight(dataframe: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    """Equilibra as classes sem duplicar linhas de treino."""

    counts = {int(row["label"]): int(row["count"]) for row in dataframe.groupBy("label").count().collect()}
    if set(counts) != {0, 1}:
        raise ValueError("O treino precisa conter as classes 0 e 1.")
    total = sum(counts.values())
    weighted = dataframe.withColumn(
        "class_weight",
        F.when(F.col("label") == 0.0, F.lit(total / (2 * counts[0]))).otherwise(
            F.lit(total / (2 * counts[1]))
        ),
    )
    return weighted, {"not_literate": counts[0], "literate": counts[1]}


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """Monta o pré-processamento comum a todos os classificadores."""

    stages: list[Any] = []
    assembled_columns: list[str] = []

    if numeric_columns:
        numeric_output = [f"__{column}_imputed" for column in numeric_columns]
        stages.append(
            Imputer(
                inputCols=numeric_columns,
                outputCols=numeric_output,
                strategy="median",
            )
        )
        assembled_columns.extend(numeric_output)

    if categorical_columns:
        index_columns = [f"__{column}_index" for column in categorical_columns]
        encoded_columns = [f"__{column}_ohe" for column in categorical_columns]
        for source, destination in zip(categorical_columns, index_columns):
            stages.append(
                StringIndexer(
                    inputCol=source,
                    outputCol=destination,
                    handleInvalid="keep",
                )
            )
        stages.append(
            OneHotEncoder(
                inputCols=index_columns,
                outputCols=encoded_columns,
                handleInvalid="keep",
            )
        )
        assembled_columns.extend(encoded_columns)

    stages.append(VectorAssembler(inputCols=assembled_columns, outputCol="features", handleInvalid="keep"))
    return Pipeline(stages=stages)


def candidate_models(num_trees: int, max_depth: int, gbt_max_iter: int) -> dict[str, Any]:
    """Retorna os equivalentes Spark ML dos modelos comparados no fluxo Pandas."""

    common_parameters = {
        "labelCol": "label",
        "featuresCol": "features",
        "weightCol": "class_weight",
        "predictionCol": "prediction",
    }
    probabilistic_parameters = {
        **common_parameters,
        "probabilityCol": "probability",
        "rawPredictionCol": "rawPrediction",
    }
    return {
        "logistic_regression": LogisticRegression(
            **probabilistic_parameters,
            maxIter=1000,
            regParam=0.0,
        ),
        "random_forest": RandomForestClassifier(
            **probabilistic_parameters,
            numTrees=num_trees,
            maxDepth=max_depth,
            featureSubsetStrategy="sqrt",
            seed=42,
        ),
        # Spark ML não possui HistGradientBoostingClassifier. GBTClassifier é
        # o classificador de árvores de gradiente equivalente disponível.
        "gradient_boosted_trees": GBTClassifier(
            **common_parameters,
            maxIter=gbt_max_iter,
            maxDepth=max_depth,
            stepSize=0.08,
            seed=42,
        ),
    }


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics_from_confusion_matrix(
    model_name: str,
    true_negative: int,
    false_positive: int,
    false_negative: int,
    true_positive: int,
    roc_auc: float,
    pr_auc: float,
) -> dict[str, Any]:
    """Converte a matriz de confusão nas métricas do relatório."""

    total = true_negative + false_positive + false_negative + true_positive
    precision_zero = safe_divide(true_negative, true_negative + false_positive)
    recall_zero = safe_divide(true_negative, true_negative + false_negative)
    precision_one = safe_divide(true_positive, true_positive + false_negative)
    recall_one = safe_divide(true_positive, true_positive + false_positive)
    return {
        "model": model_name,
        "n_samples": total,
        "accuracy": safe_divide(true_negative + true_positive, total),
        "precision_literate": precision_one,
        "recall_literate": recall_one,
        "f1_literate": safe_divide(2 * precision_one * recall_one, precision_one + recall_one),
        "precision_not_literate": precision_zero,
        "recall_not_literate": recall_zero,
        "f1_not_literate": safe_divide(2 * precision_zero * recall_zero, precision_zero + recall_zero),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": [[true_negative, false_negative], [false_positive, true_positive]],
    }


def evaluate(model_name: str, predictions: DataFrame) -> tuple[dict[str, Any], DataFrame]:
    """Calcula métricas e mantém a predição distribuída para as agregações."""

    cached = predictions.cache()
    cached.count()
    matrix = {
        (int(row["label"]), int(row["prediction"])): int(row["count"])
        for row in cached.groupBy("label", "prediction").count().collect()
    }
    true_negative = matrix.get((0, 0), 0)
    false_positive = matrix.get((1, 0), 0)
    false_negative = matrix.get((0, 1), 0)
    true_positive = matrix.get((1, 1), 0)
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    roc_auc = evaluator.setMetricName("areaUnderROC").evaluate(cached)
    pr_auc = evaluator.setMetricName("areaUnderPR").evaluate(cached)
    return (
        metrics_from_confusion_matrix(
            model_name,
            true_negative,
            false_positive,
            false_negative,
            true_positive,
            roc_auc,
            pr_auc,
        ),
        cached,
    )


def evaluate_dummy(test_frame: DataFrame, train_counts: dict[str, int]) -> dict[str, Any]:
    """Avalia o baseline que prevê a classe majoritária do treino para todos."""

    test_counts = {int(row["label"]): int(row["count"]) for row in test_frame.groupBy("label").count().collect()}
    majority_class = 0 if train_counts["not_literate"] > train_counts["literate"] else 1
    if majority_class == 0:
        true_negative = test_counts.get(0, 0)
        false_positive = test_counts.get(1, 0)
        false_negative = true_positive = 0
    else:
        false_negative = test_counts.get(0, 0)
        true_positive = test_counts.get(1, 0)
        true_negative = false_positive = 0
    return metrics_from_confusion_matrix(
        "dummy",
        true_negative,
        false_positive,
        false_negative,
        true_positive,
        roc_auc=0.5,
        pr_auc=safe_divide(test_counts.get(1, 0), sum(test_counts.values())),
    )


def class_distribution(dataframe: DataFrame) -> dict[str, Any]:
    counts = {int(row["label"]): int(row["count"]) for row in dataframe.groupBy("label").count().collect()}
    total = sum(counts.values())
    return {
        "n_samples": total,
        "not_literate": counts.get(0, 0),
        "literate": counts.get(1, 0),
        "not_literate_rate": safe_divide(counts.get(0, 0), total),
        "literate_rate": safe_divide(counts.get(1, 0), total),
    }


def aggregate_predictions(predictions: DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Gera agregações pequenas, sem salvar previsões individuais de alunos."""

    scored = (
        predictions.withColumn("actual_not_literate", (F.col("label") == 0.0).cast("int"))
        .withColumn("predicted_not_literate", (F.col("prediction") == 0.0).cast("int"))
        .withColumn("not_literate_risk", vector_to_array("probability")[0])
    )
    aggregations: dict[str, list[dict[str, Any]]] = {}
    for column in GROUP_COLUMNS:
        if column not in scored.columns:
            continue
        rows = (
            scored.groupBy(column)
            .agg(
                F.count("*").alias("records"),
                F.avg("actual_not_literate").alias("actual_not_literate_rate"),
                F.avg("predicted_not_literate").alias("predicted_not_literate_rate"),
                F.avg("not_literate_risk").alias("average_not_literate_risk"),
            )
            .orderBy(F.desc("average_not_literate_risk"), F.desc("records"))
            .collect()
        )
        aggregations[f"by_{column}"] = [
            {
                key: (float(value) if isinstance(value, float) else value)
                for key, value in row.asDict(recursive=True).items()
            }
            for row in rows
        ]
    return aggregations


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"URI S3 inválida: {uri}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def write_json(uri: str, filename: str, payload: dict[str, Any]) -> str:
    """Persiste metadados em JSON localmente ou no S3."""

    content = json.dumps(payload, ensure_ascii=False, indent=2)
    if uri.startswith("s3://"):
        import boto3

        bucket, prefix = parse_s3_uri(uri)
        key = "/".join(part for part in (prefix, filename) if part)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"))
        return f"s3://{bucket}/{key}"

    output = Path(normalize_uri(uri))
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    destination.write_text(content, encoding="utf-8")
    return str(destination)


def train(
    spark: SparkSession,
    input_uri: str,
    output_uri: str,
    num_trees: int,
    max_depth: int,
    gbt_max_iter: int,
) -> dict[str, Any]:
    """Compara modelos, seleciona o melhor e grava os artefatos Spark ML."""

    started_at = perf_counter()
    raw = spark.read.parquet(normalize_uri(input_uri))
    train_raw, test_raw, train_years, test_year = split_temporally(raw)
    train_frame, numeric_columns, categorical_columns = prepare_dataset(train_raw)
    test_frame, _, _ = prepare_dataset(test_raw)
    train_frame, train_counts = add_class_weight(train_frame)
    test_distribution = class_distribution(test_frame)

    # O pré-processamento é ajustado uma única vez e o resultado é reutilizado
    # pelos três modelos, evitando repetir indexação e one-hot encoding.
    preprocessor_model = build_preprocessor(numeric_columns, categorical_columns).fit(train_frame)
    prepared_train = preprocessor_model.transform(train_frame).cache()
    prepared_test = preprocessor_model.transform(test_frame).cache()
    prepared_train.count()
    prepared_test.count()

    all_metrics = {"dummy": evaluate_dummy(prepared_test, train_counts)}
    trained_models: dict[str, Any] = {}
    selected_predictions: DataFrame | None = None
    selected_name: str | None = None
    for model_name, estimator in candidate_models(num_trees, max_depth, gbt_max_iter).items():
        classifier_model = estimator.fit(prepared_train)
        model_metrics, predictions = evaluate(model_name, classifier_model.transform(prepared_test))
        all_metrics[model_name] = model_metrics
        trained_models[model_name] = classifier_model

        candidate_key = (
            model_metrics["recall_not_literate"],
            model_metrics["f1_literate"],
            model_metrics["pr_auc"],
        )
        if selected_name is None or candidate_key > (
            all_metrics[selected_name]["recall_not_literate"],
            all_metrics[selected_name]["f1_literate"],
            all_metrics[selected_name]["pr_auc"],
        ):
            if selected_predictions is not None:
                selected_predictions.unpersist()
            selected_name = model_name
            selected_predictions = predictions
        else:
            predictions.unpersist()

    assert selected_name is not None and selected_predictions is not None
    metrics = all_metrics[selected_name].copy()
    aggregations = aggregate_predictions(selected_predictions)

    model_path = f"{output_uri.rstrip('/')}/model"
    PipelineModel(stages=[*preprocessor_model.stages, trained_models[selected_name]]).write().overwrite().save(
        normalize_uri(model_path)
    )
    metrics.update(
        {
            "selected_model": selected_name,
            "selection_metric": "recall_not_literate",
            "test_year": test_year,
            "train_years": train_years,
            "features": numeric_columns + categorical_columns,
            "feature_types": {
                "numeric": numeric_columns,
                "categorical": categorical_columns,
            },
            "feature_transformations": {
                "one_hot_encoded": sorted(FORCED_CATEGORICAL_COLUMNS),
                "log1p": sorted(LOG1P_COLUMNS),
            },
            "training_class_distribution": train_counts,
            "test_class_distribution": test_distribution,
            "parameters": {
                "random_forest_num_trees": num_trees,
                "tree_max_depth": max_depth,
                "gradient_boosted_trees_max_iter": gbt_max_iter,
            },
            "comparison": all_metrics,
            "elapsed_seconds": round(perf_counter() - started_at, 2),
            "model_path": model_path,
        }
    )
    metrics_path = write_json(output_uri, "metrics.json", metrics)
    aggregations_path = write_json(output_uri, "aggregations.json", aggregations)
    selected_predictions.unpersist()
    prepared_train.unpersist()
    prepared_test.unpersist()
    print(
        json.dumps(
            {
                "model_path": model_path,
                "metrics_path": metrics_path,
                "aggregations_path": aggregations_path,
                "elapsed_seconds": metrics["elapsed_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return metrics


def run_glue_job() -> None:
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark.context import SparkContext

    arguments = getResolvedOptions(sys.argv, ["JOB_NAME", "BUCKET_GOLD", "BUCKET_MODEL"])
    spark_context = SparkContext.getOrCreate()
    glue_context = GlueContext(spark_context)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(arguments["JOB_NAME"], arguments)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_uri = "/".join(
        (
            arguments["BUCKET_MODEL"].rstrip("/"),
            GLUE_OUTPUT_PATH,
            f"{arguments['JOB_NAME']}-{timestamp}",
        )
    )
    train(
        spark,
        f"{arguments['BUCKET_GOLD'].rstrip('/')}/{GLUE_INPUT_PATH}",
        output_uri,
        num_trees=200,
        max_depth=12,
        gbt_max_iter=200,
    )
    job.commit()


def main() -> None:
    if "--JOB_NAME" in sys.argv:
        run_glue_job()
        return

    arguments = parse_args()
    spark = SparkSession.builder.appName("alfabetizacao-spark-ml").getOrCreate()
    try:
        train(
            spark,
            arguments.input,
            arguments.output,
            arguments.num_trees,
            arguments.max_depth,
            arguments.gbt_max_iter,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
