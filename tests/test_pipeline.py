"""Testes de sanidade: garante que colunas proibidas (leakage) nunca viram feature."""
import numpy as np
import pandas as pd

from src.modeling.features import ENGINEERED, EXCLUDE, build_pipeline, feature_columns


def _toy(n=300):
    """Base sintética mínima com as colunas que a pipeline espera."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "escola_lag_taxa_alfab": rng.random(n), "escola_lag_n_alunos": rng.integers(5, 90, n).astype(float),
        "mun_lag_alfabetizado": rng.random(n) * 100, "uf_lag_taxa_alfab": rng.random(n) * 100,
        "fundeb_por_matricula": rng.random(n) * 1e4, "mun_lag_nivel_3": rng.random(n),
        "SG_UF": rng.choice(["SP", "BA"], n), "TP_DEPENDENCIA": rng.choice([2, 3], n), "regiao": rng.choice([2, 3], n),
        "VL_PROFICIENCIA_LP": rng.random(n) * 900, "ID_ESCOLA": rng.integers(0, 20, n),
        "ID_ALUNO": np.arange(n), "CO_MUNICIPIO": 1, "CO_UF": 11, "NU_ANO_AVALIACAO": 2024,
    })
    df["IN_ALFABETIZADO"] = (df.escola_lag_taxa_alfab + rng.normal(0, .3, n) > .5).astype(int)
    df.loc[::7, "escola_lag_taxa_alfab"] = np.nan  # força imputação
    return df


def test_leakage_columns_excluded():
    num, cat = feature_columns(_toy())
    assert not (set(num + cat) & EXCLUDE)
    assert not any(c.startswith("mun_lag_nivel_") for c in num)


def test_pipeline_fits_with_missing_values():
    df = _toy()
    num, cat = feature_columns(df)
    p = build_pipeline(num, cat).fit(df[num + cat], df["IN_ALFABETIZADO"])
    proba = p.predict_proba(df[num + cat])[:, 1]
    assert proba.shape == (len(df),) and not np.isnan(proba).any()
    assert set(ENGINEERED) <= set(p[:-1].get_feature_names_out())
