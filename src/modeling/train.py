"""Treino com validação temporal: treina em 2024, valida por GroupKFold(escola), testa em 2025."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from src.config import GOLD, MODELS, REPORTS, SEED, TARGET, TEST_YEAR, TRAIN_YEAR
from src.evaluation.metrics import compute_metrics
from src.modeling.features import build_pipeline, feature_columns


def load_split():
    """Carrega a Gold e separa treino (2024) e teste fora do tempo (2025)."""
    df = pd.read_parquet(GOLD / "alunos_gold.parquet")
    num, cat = feature_columns(df)
    tr, te = df[df.NU_ANO_AVALIACAO == TRAIN_YEAR], df[df.NU_ANO_AVALIACAO == TEST_YEAR]
    return tr, te, num, cat


def sample_by_school(df: pd.DataFrame, n_schools: int) -> pd.DataFrame:
    """Subamostra por ESCOLA (não por aluno) para manter grupos íntegros no CV."""
    rng = np.random.default_rng(SEED)
    schools = df["ID_ESCOLA"].unique()
    keep = rng.choice(schools, size=min(n_schools, len(schools)), replace=False)
    return df[df["ID_ESCOLA"].isin(keep)]


def main():
    MODELS.mkdir(exist_ok=True), REPORTS.mkdir(exist_ok=True)
    tr, te, num, cat = load_split()
    X_tr, y_tr, X_te, y_te = tr[num + cat], tr[TARGET], te[num + cat], te[TARGET]
    results = {}

    # 1) Baselines: prior (piso) e regressão logística (modelo linear interpretável)
    dummy = build_pipeline(num, cat, DummyClassifier(strategy="prior")).fit(X_tr, y_tr)
    results["dummy"] = compute_metrics(y_te, dummy.predict_proba(X_te)[:, 1])
    sub = sample_by_school(tr, 8000)
    lr = build_pipeline(num, cat, LogisticRegression(max_iter=500, C=0.5)).fit(sub[num + cat], sub[TARGET])
    results["logistic_regression"] = compute_metrics(y_te, lr.predict_proba(X_te)[:, 1])

    # 2) Modelo principal: HistGradientBoosting com busca aleatória + GroupKFold por escola
    cv_df = sample_by_school(tr, 5000)
    search = RandomizedSearchCV(
        build_pipeline(num, cat),
        param_distributions={
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_depth": [4, 6, 8, None],
            "clf__max_leaf_nodes": [15, 31, 63],
            "clf__min_samples_leaf": [50, 200, 1000],
            "clf__l2_regularization": [0.0, 1.0, 10.0],
            "clf__max_iter": [200, 400],
        },
        n_iter=10, scoring="roc_auc", cv=GroupKFold(n_splits=3), random_state=SEED, n_jobs=1, verbose=1,
    )
    search.fit(cv_df[num + cat], cv_df[TARGET], groups=cv_df["ID_ESCOLA"])
    # Refit do melhor conjunto de hiperparâmetros em TODO o 2024
    best = build_pipeline(num, cat, HistGradientBoostingClassifier(random_state=SEED, early_stopping=True,
                          **{k.split("__")[1]: v for k, v in search.best_params_.items()}))
    best.fit(X_tr, y_tr)
    proba = best.predict_proba(X_te)[:, 1]
    results["hist_gradient_boosting"] = compute_metrics(y_te, proba)
    results["cv_roc_auc_mean_std"] = [float(search.best_score_),
                                      float(search.cv_results_["std_test_score"][search.best_index_])]
    results["best_params"] = search.best_params_

    # 3) Persistência: modelo, predições de teste e métricas
    joblib.dump(best, MODELS / "hgb_pipeline.joblib")
    joblib.dump(lr, MODELS / "logreg_pipeline.joblib")
    te[["ID_ALUNO", "ID_ESCOLA", "CO_MUNICIPIO", "SG_UF", "TP_DEPENDENCIA", TARGET]].assign(proba=proba).to_parquet(GOLD / "predicoes_teste.parquet")
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
