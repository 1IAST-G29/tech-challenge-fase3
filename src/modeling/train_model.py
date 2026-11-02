"""Treina um classificador para prever o status de alfabetização do aluno.

Uso local:
    python src/modeling/train_model.py \
        --input local-data/gold/ml/ml_dataset_alunos \
        --output artifacts/model

Uso no AWS Glue:
    --JOB_NAME <nome> \
    --BUCKET_GOLD s3://<bucket>/gold \
    --BUCKET_MODEL s3://<bucket>/modeling
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
from urllib.parse import urlparse

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "alfabetizado"
GLUE_INPUT_PATH = "ml/ml_dataset_alunos"
GLUE_OUTPUT_PATH = "alfabetizacao/runs"
INTERPRETABILITY_SAMPLE_SIZE = 5_000
TOP_FEATURES = 20
FORBIDDEN_COLUMNS = {
    "ano",
    "proficiencia",
    "vl_proficiencia_lp",
    "id_aluno",
    "id_escola",
    "id_municipio",
    "state_code",
    "co_municipio",
    "co_uf",
}


def parse_args() -> argparse.Namespace:
    """Lê os caminhos da execução local."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Diretório Parquet da ml_dataset_alunos.")
    parser.add_argument("--output", required=True, help="Diretório dos artefatos do treinamento.")
    return parser.parse_args()


def read_dataset(input_path: str | Path) -> pd.DataFrame:
    """Lê a base Parquet local e valida a presença do alvo."""

    input_path_str = str(input_path)
    if input_path_str.startswith("s3://"):
        raise ValueError(
            "A leitura S3 ainda não está habilitada neste script. "
            "Baixe a Gold para Parquet local ou monte o bucket no ambiente de treino."
        )

    dataframe = pd.read_parquet(input_path_str.removeprefix("file://"))
    if TARGET not in dataframe.columns:
        raise ValueError(f"A base precisa conter a coluna alvo {TARGET!r}.")
    if "ano" in dataframe.columns:
        dataframe["ano"] = pd.to_numeric(dataframe["ano"], errors="coerce")
    return dataframe


def read_glue_dataset(spark: Any, bucket_gold: str) -> pd.DataFrame:
    """Lê a Gold no S3 com Spark e a entrega ao Scikit-learn como Pandas."""

    dataframe = spark.read.parquet(f"{bucket_gold.rstrip('/')}/{GLUE_INPUT_PATH}").toPandas()
    if TARGET not in dataframe.columns:
        raise ValueError(f"A base precisa conter a coluna alvo {TARGET!r}.")
    dataframe["ano"] = pd.to_numeric(dataframe["ano"], errors="coerce")
    return dataframe


def split_dataset(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa o último ano para teste, evitando mistura temporal entre treino e teste."""

    if "ano" not in dataframe.columns:
        raise ValueError("A base precisa conter a coluna ano para validação temporal.")

    years = sorted(dataframe["ano"].dropna().unique())
    if len(years) < 2:
        raise ValueError("É necessário ter pelo menos dois anos para o teste temporal.")

    test_year = years[-1]
    train = dataframe[dataframe["ano"] < test_year].copy()
    test = dataframe[dataframe["ano"] == test_year].copy()
    if train.empty or test.empty:
        raise ValueError("A separação temporal produziu treino ou teste vazio.")
    return train, test


def prepare_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Remove alvo, identificadores e colunas proibidas antes do treinamento."""

    target = pd.to_numeric(dataframe[TARGET], errors="coerce")
    valid_target = target.isin([0, 1])
    dataframe = dataframe.loc[valid_target].copy()
    target = target.loc[valid_target].astype(int)

    columns_to_drop = (FORBIDDEN_COLUMNS | {TARGET}) & set(dataframe.columns)
    features = dataframe.drop(columns=list(columns_to_drop))
    return features, target


def build_pipeline(features: pd.DataFrame, estimator: Any) -> Pipeline:
    """Cria o pipeline de preparação seguido do classificador informado."""

    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    if isinstance(estimator, HistGradientBoostingClassifier):
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
    else:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0 if isinstance(estimator, HistGradientBoostingClassifier) else 0.3,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def candidate_models() -> dict[str, Any]:
    """Retorna os quatro modelos comparados no treinamento."""

    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        ),
    }


def evaluate(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    """Calcula métricas de classificação e dá destaque à classe não alfabetizada."""

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "n_samples": int(len(target)),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision_literate": float(precision_score(target, predictions, pos_label=1, zero_division=0)),
        "precision_not_literate": float(
            precision_score(target, predictions, pos_label=0, zero_division=0)
        ),
        "recall_literate": float(recall_score(target, predictions, pos_label=1, zero_division=0)),
        "f1_literate": float(f1_score(target, predictions, pos_label=1, zero_division=0)),
        "recall_not_literate": float(recall_score(target, predictions, pos_label=0, zero_division=0)),
        "f1_not_literate": float(f1_score(target, predictions, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "confusion_matrix": confusion_matrix(target, predictions).tolist(),
    }


def class_distribution(target: pd.Series) -> dict[str, Any]:
    """Resume a proporção das duas classes do alvo."""

    counts = target.value_counts().to_dict()
    total = len(target)
    return {
        "n_samples": int(total),
        "not_literate": int(counts.get(0, 0)),
        "literate": int(counts.get(1, 0)),
        "not_literate_rate": float(counts.get(0, 0) / total),
        "literate_rate": float(counts.get(1, 0) / total),
    }


def explain_confusion_matrix(matrix: list[list[int]]) -> dict[str, int]:
    """Traduz a matriz de confusão para os resultados relevantes à política pública."""

    return {
        "not_literate_correctly_identified": int(matrix[0][0]),
        "not_literate_missed": int(matrix[0][1]),
        "unnecessary_not_literate_alerts": int(matrix[1][0]),
        "literate_correctly_identified": int(matrix[1][1]),
    }


def calculate_feature_importance(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    """Calcula importância por permutação em uma amostra do conjunto de teste."""

    sample_size = min(INTERPRETABILITY_SAMPLE_SIZE, len(features))
    sampled_features = features.sample(n=sample_size, random_state=42)
    sampled_target = target.loc[sampled_features.index]
    result = permutation_importance(
        model,
        sampled_features,
        sampled_target,
        scoring="average_precision",
        n_repeats=3,
        n_jobs=1,
        random_state=42,
    )
    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return {
        "method": "permutation_importance",
        "scoring": "average_precision",
        "sample_size": int(sample_size),
        "top_features": json.loads(importance.head(TOP_FEATURES).to_json(orient="records")),
    }


def aggregate_predictions(
    test_frame: pd.DataFrame,
    target: pd.Series,
    predictions: Any,
    probabilities: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Gera resultados agregados, sem persistir previsões individuais de alunos."""

    scored = test_frame.copy()
    scored["actual_not_literate"] = (target == 0).astype(int)
    scored["predicted_not_literate"] = (predictions == 0).astype(int)
    scored["not_literate_risk"] = 1 - probabilities

    aggregations = {}
    for column in ("city_code", "state_abbreviation", "education_network_type"):
        if column not in scored.columns:
            continue
        grouped = (
            scored.groupby(column, dropna=False)
            .agg(
                records=("actual_not_literate", "size"),
                actual_not_literate_rate=("actual_not_literate", "mean"),
                predicted_not_literate_rate=("predicted_not_literate", "mean"),
                average_not_literate_risk=("not_literate_risk", "mean"),
            )
            .reset_index()
            .sort_values(["average_not_literate_risk", "records"], ascending=[False, False])
        )
        aggregations[f"by_{column}"] = json.loads(grouped.to_json(orient="records"))
    return aggregations


def build_training_report(
    model: Pipeline,
    train_frame: pd.DataFrame,
    train_target: pd.Series,
    test_frame: pd.DataFrame,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    selected_metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Monta os resultados adicionais destinados à apresentação do melhor modelo."""

    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]
    return (
        {
            "target": {
                "column": TARGET,
                "literate_value": 1,
                "not_literate_value": 0,
                "objective": "Prever se um aluno será alfabetizado ou não.",
            },
            "validation_strategy": {
                "type": "temporal_holdout",
                "train_years": sorted(
                    int(year) for year in train_frame["ano"].dropna().unique()
                ),
                "test_year": int(test_frame["ano"].iloc[0]),
            },
            "class_distribution": {
                "train": class_distribution(train_target),
                "test": class_distribution(test_target),
            },
            "confusion_matrix_interpretation": explain_confusion_matrix(
                selected_metrics["confusion_matrix"]
            ),
            "feature_importance": calculate_feature_importance(model, test_features, test_target),
        },
        aggregate_predictions(test_frame, test_target, predictions, probabilities),
    )


def train_dataframe(dataframe: pd.DataFrame, output_path: str | Path) -> dict[str, Any]:
    """Treina, compara, seleciona e salva o melhor modelo e suas métricas."""

    train_frame, test_frame = split_dataset(dataframe)
    train_features, train_target = prepare_features(train_frame)
    test_features, test_target = prepare_features(test_frame)

    if train_target.nunique() < 2:
        raise ValueError("O treino precisa conter as duas classes do alvo.")

    all_metrics: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Pipeline] = {}
    for model_name, estimator in candidate_models().items():
        model = build_pipeline(train_features, estimator)
        model.fit(train_features, train_target)
        model_metrics = evaluate(model, test_features, test_target)
        model_metrics["model"] = model_name
        all_metrics[model_name] = model_metrics
        trained_models[model_name] = model

    best_name = max(
        all_metrics,
        key=lambda name: (
            all_metrics[name]["recall_not_literate"],
            all_metrics[name]["f1_literate"],
            all_metrics[name]["pr_auc"],
        ),
    )
    metrics = all_metrics[best_name].copy()
    metrics.update(
        {
            "selected_model": best_name,
            "selection_metric": "recall_not_literate",
            "test_year": int(test_frame["ano"].iloc[0]),
            "train_years": sorted(int(year) for year in train_frame["ano"].dropna().unique()),
            "features": train_features.columns.tolist(),
            "comparison": all_metrics,
        }
    )

    presentation_report, aggregations = build_training_report(
        trained_models[best_name],
        train_frame,
        train_target,
        test_frame,
        test_features,
        test_target,
        metrics,
    )
    metrics["presentation"] = presentation_report

    artifacts = Path(output_path)
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models[best_name], artifacts / "model.joblib")
    metrics_json = json.dumps(metrics, ensure_ascii=False, indent=2)
    metrics_path = artifacts / "metrics.json"
    metrics_path.write_text(metrics_json, encoding="utf-8")
    (artifacts / "aggregations.json").write_text(
        json.dumps(aggregations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def train(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Executa o treinamento local a partir de uma pasta Parquet."""

    return train_dataframe(read_dataset(input_path), output_path)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Separa um URI S3 em bucket e prefixo, validando o formato recebido."""

    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"URI S3 inválido: {uri}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def upload_artifacts(local_directory: Path, destination_uri: str) -> list[str]:
    """Envia o modelo e as métricas para S3 ou para ``file://`` nos testes locais."""

    if destination_uri.startswith("file://"):
        destination = Path(destination_uri.removeprefix("file://"))
        destination.mkdir(parents=True, exist_ok=True)
        uploaded_uris = []
        for filename in ("model.joblib", "metrics.json", "aggregations.json"):
            target = destination / filename
            shutil.copy2(local_directory / filename, target)
            uploaded_uris.append(target.as_uri())
        return uploaded_uris

    import boto3

    bucket, prefix = parse_s3_uri(destination_uri)
    s3 = boto3.client("s3")
    uploaded_uris = []
    for filename in ("model.joblib", "metrics.json", "aggregations.json"):
        key = "/".join(part for part in (prefix, filename) if part)
        s3.upload_file(str(local_directory / filename), bucket, key)
        uploaded_uris.append(f"s3://{bucket}/{key}")
    return uploaded_uris


def run_glue_job() -> None:
    """Executa o treinamento como AWS Glue Spark Job e publica os artefatos no S3."""

    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark.context import SparkContext

    arguments = getResolvedOptions(sys.argv, ["JOB_NAME", "BUCKET_GOLD", "BUCKET_MODEL"])
    spark_context = SparkContext.getOrCreate()
    glue_context = GlueContext(spark_context)
    job = Job(glue_context)
    job.init(arguments["JOB_NAME"], arguments)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination_uri = "/".join(
        (
            arguments["BUCKET_MODEL"].rstrip("/"),
            GLUE_OUTPUT_PATH,
            f"{arguments['JOB_NAME']}-{timestamp}",
        )
    )
    with tempfile.TemporaryDirectory(prefix="alfabetizacao-model-") as temporary_directory:
        artifacts_directory = Path(temporary_directory)
        dataframe = read_glue_dataset(glue_context.spark_session, arguments["BUCKET_GOLD"])
        metrics = train_dataframe(dataframe, artifacts_directory)
        uploaded_uris = upload_artifacts(artifacts_directory, destination_uri)

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Artefatos do modelo gravados em: {', '.join(uploaded_uris)}", flush=True)
    job.commit()


def main() -> None:
    """Executa localmente ou pelo contrato de argumentos de um AWS Glue Job."""

    if "--JOB_NAME" in sys.argv:
        run_glue_job()
        return

    arguments = parse_args()
    metrics = train(arguments.input, arguments.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
