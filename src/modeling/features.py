"""Definição das features e da pipeline sklearn (pré-processamento integrado ao modelo)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import SEED

# Categóricas de baixa cardinalidade (27 UFs, 2 redes, 5 regiões)
CATEGORICAL = ["SG_UF", "TP_DEPENDENCIA", "regiao"]
# Nunca entram: alvo, proficiência (define o alvo), IDs e ano
EXCLUDE = {"IN_ALFABETIZADO", "VL_PROFICIENCIA_LP", "ID_ALUNO", "ID_ESCOLA", "CO_MUNICIPIO",
           "CO_UF", "NU_ANO_AVALIACAO"}
# Distribuição por níveis só existe a partir do lag 2024 -> drift treino/teste, então fora
EXCLUDE_PREFIX = ("mun_lag_nivel_",)


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa colunas numéricas e categóricas, removendo as proibidas (anti-leakage)."""
    ok = [c for c in df.columns if c not in EXCLUDE and not c.startswith(EXCLUDE_PREFIX)]
    cat = [c for c in CATEGORICAL if c in ok]
    num = [c for c in ok if c not in cat]
    return num, cat


def add_engineered(X: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering sem estado (seguro dentro de CV): hiatos e razões históricas."""
    X = X.copy()
    # Escola vs município: escola acima/abaixo do contexto municipal (taxa em pontos percentuais)
    X["gap_escola_mun"] = X["escola_lag_taxa_alfab"] * 100 - X["mun_lag_alfabetizado"]
    # Município vs UF
    X["gap_mun_uf"] = X["mun_lag_alfabetizado"] - X["uf_lag_taxa_alfab"]
    # Escalas muito assimétricas -> log
    X["log_escola_lag_n"] = np.log1p(X["escola_lag_n_alunos"])
    X["log_fundeb_por_mat"] = np.log1p(X["fundeb_por_matricula"])
    return X


ENGINEERED = ["gap_escola_mun", "gap_mun_uf", "log_escola_lag_n", "log_fundeb_por_mat"]


def engineered_names(_, names):
    """Nomes de saída do passo de engenharia (função nomeada para ser serializável)."""
    return list(names) + ENGINEERED


def build_pipeline(num: list[str], cat: list[str], model=None) -> Pipeline:
    """Pipeline completa: engenharia -> imputação/escala/encoding -> modelo."""
    pre = ColumnTransformer([
        # Mediana é robusta a outliers; add_indicator preserva a informação "faltava"
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num + ENGINEERED),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ], verbose_feature_names_out=False)
    model = model or HistGradientBoostingClassifier(random_state=SEED, early_stopping=True)
    return Pipeline([("fe", FunctionTransformer(add_engineered, feature_names_out=engineered_names)),
                     ("pre", pre), ("clf", model)])
