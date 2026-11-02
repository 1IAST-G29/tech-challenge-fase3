"""Treina um classificador para prever o status de alfabetização do aluno.

Uso:
    python src/modeling/train_model.py \
        --input data/gold/ml_dataset_alunos \
        --output artifacts/model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "alfabetizado"
FORBIDDEN_COLUMNS = {
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
    """Lê os caminhos informados na linha de comando."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Diretório Parquet da ml_dataset_alunos")
    parser.add_argument("--output", required=True, help="Diretório dos artefatos do treinamento")
    return parser.parse_args()


def read_dataset(input_path: str) -> pd.DataFrame:
    """Lê a base Parquet local e valida a presença do alvo."""

    if input_path.startswith("s3://"):
        raise ValueError(
            "A leitura S3 ainda não está habilitada neste script. "
            "Baixe a Gold para Parquet local ou monte o bucket no ambiente de treino."
        )

    dataframe = pd.read_parquet(input_path.removeprefix("file://"))
    if TARGET not in dataframe.columns:
        raise ValueError(f"A base precisa conter a coluna alvo {TARGET!r}.")
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
        "recall_literate": float(recall_score(target, predictions, pos_label=1, zero_division=0)),
        "f1_literate": float(f1_score(target, predictions, pos_label=1, zero_division=0)),
        "recall_not_literate": float(recall_score(target, predictions, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "confusion_matrix": confusion_matrix(target, predictions).tolist(),
    }


def train(input_path: str, output_path: str) -> dict[str, Any]:
    """Treina, compara, seleciona e salva o melhor modelo e suas métricas."""

    dataframe = read_dataset(input_path)
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

    artifacts = Path(output_path)
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models[best_name], artifacts / "model.joblib")
    (artifacts / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    """Executa o treinamento a partir dos argumentos da linha de comando."""

    arguments = parse_args()
    metrics = train(arguments.input, arguments.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
