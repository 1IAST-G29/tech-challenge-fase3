"""Análise exploratória: gera figuras e tabelas que embasam as decisões de modelagem."""
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import GOLD, REPORTS, TARGET
from src.visualization.plots import barh, heatmap, hist_by


def main():
    df = pd.read_parquet(GOLD / "alunos_gold.parquet")
    out = {}
    # Balanceamento do alvo por ano e por rede
    out["alvo_por_ano"] = df.groupby("NU_ANO_AVALIACAO")[TARGET].mean().round(4).to_dict()
    out["alvo_por_rede"] = df.groupby("TP_DEPENDENCIA")[TARGET].mean().round(4).to_dict()
    out["alvo_por_regiao"] = df.groupby("regiao")[TARGET].mean().round(4).to_dict()
    # Demonstração de leakage: a proficiência separa quase perfeitamente o alvo -> excluída
    s = df.dropna(subset=["VL_PROFICIENCIA_LP"]).sample(200000, random_state=0)
    out["auc_proficiencia_mesmo_ano_(LEAKAGE)"] = float(roc_auc_score(s[TARGET], s["VL_PROFICIENCIA_LP"]))
    out["ponto_de_corte_proficiencia"] = float(s[s[TARGET] == 1]["VL_PROFICIENCIA_LP"].min())
    # Taxa por UF
    uf = df.groupby("SG_UF")[TARGET].mean().sort_values()
    barh(uf, "Taxa de alfabetização por UF (2024-2025)", "eda_taxa_por_uf.png", top=27)
    # Correlação das features numéricas históricas com o alvo (Spearman em amostra)
    num = df.select_dtypes("number").drop(columns=["ID_ALUNO", "ID_ESCOLA", "CO_MUNICIPIO", "CO_UF", "NU_ANO_AVALIACAO", "VL_PROFICIENCIA_LP"])
    num = num[[c for c in num.columns if not c.startswith("mun_lag_nivel")]].sample(300000, random_state=0)
    corr = num.corr(method="spearman")
    corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).to_csv(REPORTS / "correlacao_alvo.csv", header=["spearman"])
    barh(corr[TARGET].drop(TARGET), "Correlação de Spearman com IN_ALFABETIZADO", "eda_correlacao_alvo.png", top=25)
    top = corr[TARGET].drop(TARGET).abs().sort_values().tail(18).index.tolist() + [TARGET]
    heatmap(corr.loc[top, top], "Correlações (top 18 + alvo)", "eda_heatmap.png")
    hist_by(df.sample(300000, random_state=0), "escola_lag_taxa_alfab", TARGET, "Taxa da escola no ano anterior vs alvo", "eda_hist_escola_lag.png")
    hist_by(df.sample(300000, random_state=0), "mun_lag_alfabetizado", TARGET, "Taxa do município no ano anterior vs alvo", "eda_hist_mun_lag.png")
    # Valores faltantes por ano (guia a estratégia de imputação)
    miss = df.groupby("NU_ANO_AVALIACAO").apply(lambda g: g.isna().mean()).T
    miss[(miss > 0).any(axis=1)].round(3).to_csv(REPORTS / "faltantes_por_ano.csv")
    (REPORTS / "eda_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
