"""Aplicação estratégica: risco municipal, clusters regionais e previsão de municípios abaixo da meta."""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import GOLD, REPORTS, SEED, TARGET, TEST_YEAR, TRAIN_YEAR
from src.visualization.plots import barh

# ATENÇÃO: limiar assumido (as metas oficiais municipais estão no BigQuery, fora do repositório).
# Troque pelo valor real da meta quando a tabela meta_alfabetizacao_municipio for integrada.
META_LIMIAR = 0.60
MIN_ALUNOS = 30  # municípios com poucos alunos têm taxa instável
FEATS = ["mun_lag_alfabetizado", "mun_lag_vl_media", "uf_lag_taxa_alfab", "fundeb_total", "fundeb_por_matricula",
         "cen_rural", "cen_alunos_por_turma_ai", "cen_alunos_por_docente_ai", "cen_IN_INTERNET",
         "cen_IN_BIBLIOTECA", "cen_IN_ALIMENTACAO", "cen_pct_preta_parda_indigena", "escola_lag_taxa_alfab",
         "cen_mat_fund_ai_por_escola", "regiao"]


def municipal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega o nível aluno para município x ano (features = média; alvo = taxa observada)."""
    g = df.groupby(["CO_MUNICIPIO", "NU_ANO_AVALIACAO"])
    t = g[FEATS].mean()
    t["n_alunos"] = g.size()
    t["taxa_real"] = g[TARGET].mean()
    t["SG_UF"] = g["SG_UF"].first()
    return t.reset_index()


def main():
    df = pd.read_parquet(GOLD / "alunos_gold.parquet")
    mt = municipal_table(df)
    mt = mt[mt.n_alunos >= MIN_ALUNOS]
    tr, te = mt[mt.NU_ANO_AVALIACAO == TRAIN_YEAR], mt[mt.NU_ANO_AVALIACAO == TEST_YEAR].copy()
    out = {}

    # 1) Risco educacional 2025: probabilidade média prevista pelo modelo de aluno (fora do tempo)
    pred = pd.read_parquet(GOLD / "predicoes_teste.parquet")
    p = pred.groupby("CO_MUNICIPIO").proba.mean().rename("prob_alfab_prevista")
    te = te.merge(p, on="CO_MUNICIPIO", how="left")
    te["risco_educacional"] = 1 - te["prob_alfab_prevista"]
    out["corr_prevista_vs_real_municipio"] = float(te[["prob_alfab_prevista", "taxa_real"]].corr().iloc[0, 1])
    cols = ["CO_MUNICIPIO", "SG_UF", "n_alunos", "taxa_real", "prob_alfab_prevista", "risco_educacional"]
    te.sort_values("risco_educacional", ascending=False)[cols].head(50).to_csv(REPORTS / "top50_municipios_risco.csv", index=False)

    # 2) Clusters de municípios com perfis semelhantes (K por silhouette)
    X = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(te[FEATS])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(2500, len(X)), replace=False)
    scores = {k: silhouette_score(X[idx], KMeans(k, n_init=5, random_state=SEED).fit_predict(X[idx])) for k in range(3, 8)}
    k = max(scores, key=scores.get)
    te["cluster"] = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
    prof = te.groupby("cluster")[FEATS + ["taxa_real", "n_alunos"]].mean().round(3)
    prof["n_municipios"] = te.groupby("cluster").size()
    prof.to_csv(REPORTS / "perfil_clusters.csv")
    te[["CO_MUNICIPIO", "SG_UF", "cluster", "taxa_real", "risco_educacional"]].to_csv(REPORTS / "municipios_clusters.csv", index=False)
    out["kmeans_silhouette"] = {str(a): float(b) for a, b in scores.items()}
    out["k_escolhido"] = int(k)

    # 3) Municípios que podem não atingir a meta: treina com 2024 (lag 2023), testa em 2025 (lag 2024)
    ytr, yte = (tr.taxa_real < META_LIMIAR).astype(int), (te.taxa_real < META_LIMIAR).astype(int)
    models = {
        "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)),
        "hgb": make_pipeline(HistGradientBoostingClassifier(learning_rate=0.05, max_depth=4, random_state=SEED)),
    }
    out["meta_limiar_assumido"] = META_LIMIAR
    out["taxa_municipios_abaixo_meta"] = {"2024": float(ytr.mean()), "2025": float(yte.mean())}
    for name, m in models.items():
        m.fit(tr[FEATS], ytr)
        out[f"auc_abaixo_meta_{name}"] = float(roc_auc_score(yte, m.predict_proba(te[FEATS])[:, 1]))
    te["prob_abaixo_meta"] = models["hgb"].predict_proba(te[FEATS])[:, 1]
    te.sort_values("prob_abaixo_meta", ascending=False)[["CO_MUNICIPIO", "SG_UF", "n_alunos", "taxa_real", "prob_abaixo_meta"]].head(50).to_csv(
        REPORTS / "top50_municipios_abaixo_meta.csv", index=False)
    # Taxa média de risco por UF para o resumo executivo
    uf = te.groupby("SG_UF").agg(taxa_real=("taxa_real", "mean"), risco=("risco_educacional", "mean")).sort_values("risco")
    uf.to_csv(REPORTS / "risco_por_uf.csv")
    barh(uf["risco"], "Risco educacional médio por UF (2025, 1 - prob. prevista)", "risco_por_uf.png", top=27)
    (REPORTS / "municipal_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)); print(prof)


if __name__ == "__main__":
    main()
