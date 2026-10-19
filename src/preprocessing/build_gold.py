"""Constrói a camada Gold local (aluno e município) a partir dos dados brutos.

Princípio anti-leakage: todo atributo derivado do resultado da avaliação vem do
ANO ANTERIOR (lag) ao do aluno. Nada calculado a partir do alvo do mesmo ano entra.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

from src.config import CENSO_COLS, CENSO_FILES, GOLD, RAW, YEARS

STUDENT_COLS = ["NU_ANO_AVALIACAO", "SG_UF", "CO_UF", "ID_ALUNO", "ID_ESCOLA",
                "TP_DEPENDENCIA", "CO_MUNICIPIO", "VL_PROFICIENCIA_LP", "IN_ALFABETIZADO"]


def _read(path, **kw):
    """Lê CSV do INEP/Censo (separador ';' e encoding latin-1)."""
    return pd.read_csv(path, sep=";", encoding="latin-1", **kw)


def load_students() -> pd.DataFrame:
    """Empilha os alunos de 2023-2025 mantendo só colunas necessárias."""
    frames = [_read(RAW / "alfabetizacao_inep" / f"{y}_aluno.csv", usecols=STUDENT_COLS) for y in YEARS]
    df = pd.concat(frames, ignore_index=True)
    # Tipos inteiros enxutos (2025 vem como float por causa de NaN)
    df = df.dropna(subset=["ID_ESCOLA", "CO_MUNICIPIO", "IN_ALFABETIZADO"])
    for c in ["ID_ESCOLA", "CO_MUNICIPIO", "TP_DEPENDENCIA", "IN_ALFABETIZADO", "CO_UF"]:
        df[c] = df[c].astype("int64")
    return df


def school_lag(students: pd.DataFrame) -> pd.DataFrame:
    """Histórico da escola: taxa/proficiência média do ano Y vira feature do ano Y+1."""
    g = students.groupby(["ID_ESCOLA", "NU_ANO_AVALIACAO"]).agg(
        escola_lag_taxa_alfab=("IN_ALFABETIZADO", "mean"),
        escola_lag_prof_media=("VL_PROFICIENCIA_LP", "mean"),
        escola_lag_prof_dp=("VL_PROFICIENCIA_LP", "std"),
        escola_lag_n_alunos=("IN_ALFABETIZADO", "size"),
    ).reset_index()
    g["NU_ANO_AVALIACAO"] += 1  # desloca para o ano seguinte
    return g


def municipality_lag() -> pd.DataFrame:
    """Indicadores municipais do INEP (por rede) deslocados em +1 ano."""
    frames = []
    for y in YEARS:
        m = _read(RAW / "alfabetizacao_inep" / f"{y}_municipio.csv")
        frames.append(m.drop(columns=["NO_MUNICIPIO", "SG_UF", "TP_SERIE", "CO_UF"]))
    m = pd.concat(frames, ignore_index=True)
    # Renomeia para deixar explícito que é histórico municipal
    ren = {c: f"mun_lag_{c.lower().replace('pc_aluno_', '').replace('_lp', '')}"
           for c in m.columns if c.startswith(("PC_", "VL_"))}
    m = m.rename(columns=ren)
    m["NU_ANO_AVALIACAO"] += 1
    return m.rename(columns={"ID_TIPO_REDE": "TP_DEPENDENCIA"})


def state_lag() -> pd.DataFrame:
    """Taxa estadual (rede pública = 5) do ano anterior."""
    frames = [_read(RAW / "alfabetizacao_inep" / f"{y}_estado.csv") for y in YEARS]
    e = pd.concat(frames, ignore_index=True)
    e = e[e["ID_TIPO_REDE"] == 5][["NU_ANO_AVALIACAO", "CO_UF", "PC_ALUNO_ALFABETIZADO", "VL_MEDIA_LP"]]
    e = e.rename(columns={"PC_ALUNO_ALFABETIZADO": "uf_lag_taxa_alfab", "VL_MEDIA_LP": "uf_lag_prof_media"})
    e["NU_ANO_AVALIACAO"] += 1
    return e


def census() -> pd.DataFrame:
    """Censo Escolar do ano Y-1 agregado por município x rede (ID_ESCOLA é anonimizado).

    O ID_ESCOLA da avaliação (60000xxx) não tem tabela de correspondência com o
    CO_ENTIDADE do Censo; por isso a granularidade mais fina possível é município x rede.
    """
    frames = []
    for censo_year, path in CENSO_FILES.items():
        c = _read(path, usecols=CENSO_COLS + ["CO_MUNICIPIO", "TP_DEPENDENCIA", "TP_SITUACAO_FUNCIONAMENTO"])
        c["NU_ANO_AVALIACAO"] = censo_year + 1
        frames.append(c)
    c = pd.concat(frames, ignore_index=True)
    # Só escolas públicas em atividade que ofertam anos iniciais (as que são avaliadas)
    c = c[(c["TP_SITUACAO_FUNCIONAMENTO"] == 1) & (c["QT_MAT_FUND_AI"] > 0) & c["TP_DEPENDENCIA"].isin([2, 3])]
    c["rural"] = (c["TP_LOCALIZACAO"] == 2).astype(float)
    den = (c["QT_MAT_BAS"] - c["QT_MAT_BAS_ND"]).replace(0, np.nan)
    c["pct_preta_parda_indigena"] = (c["QT_MAT_BAS_PRETA"] + c["QT_MAT_BAS_PARDA"] + c["QT_MAT_BAS_INDIGENA"]) / den
    flags = [col for col in CENSO_COLS if col.startswith("IN_")] + ["rural", "pct_preta_parda_indigena"]
    keys = ["CO_MUNICIPIO", "TP_DEPENDENCIA", "NU_ANO_AVALIACAO"]
    # Proporção de escolas com cada infraestrutura (média de flags 0/1)
    agg = c.groupby(keys)[flags].mean().add_prefix("cen_")
    # Totais para razões ponderadas (aluno por turma/docente) e porte
    tot = c.groupby(keys)[["QT_MAT_FUND_AI", "QT_DOC_FUND_AI", "QT_TUR_FUND_AI", "QT_SALAS_UTILIZADAS",
                           "QT_PROF_PEDAGOGIA", "QT_PROF_COORDENADOR", "QT_MAT_BAS"]].sum()
    out = agg.join(tot[[]])
    out["cen_n_escolas"] = c.groupby(keys).size()
    out["cen_alunos_por_turma_ai"] = tot["QT_MAT_FUND_AI"] / tot["QT_TUR_FUND_AI"].replace(0, np.nan)
    out["cen_alunos_por_docente_ai"] = tot["QT_MAT_FUND_AI"] / tot["QT_DOC_FUND_AI"].replace(0, np.nan)
    out["cen_alunos_por_sala"] = tot["QT_MAT_FUND_AI"] / tot["QT_SALAS_UTILIZADAS"].replace(0, np.nan)
    out["cen_mat_fund_ai_por_escola"] = tot["QT_MAT_FUND_AI"] / out["cen_n_escolas"]
    out["cen_pedagogos_por_escola"] = tot["QT_PROF_PEDAGOGIA"] / out["cen_n_escolas"]
    out["mun_mat_total"] = tot["QT_MAT_BAS"]
    return out.reset_index()


def _norm(s: str) -> str:
    """Normaliza nome de município (sem acento, minúsculo) para o join do FUNDEB."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fundeb_lag(census_df: pd.DataFrame, students: pd.DataFrame) -> pd.DataFrame:
    """FUNDEB anual por município (soma dos meses do ano Y-1) e valor por matrícula."""
    f = _read(RAW / "fundeb" / "fundeb-por-municipio.csv", dtype=str)
    f.columns = ["COD_MUN", "NO_MUN", "SG_UF", "NO_MUN_UF", "MES"] + list(f.columns[5:])
    year_cols = [c for c in f.columns if c.isdigit()]
    # Converte "271.775,97" -> float
    for c in year_cols:
        f[c] = pd.to_numeric(f[c].str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
    f = f.dropna(subset=["MES"])
    anual = f.groupby(["SG_UF", "NO_MUN"])[year_cols].sum(min_count=1).reset_index()
    anual["chave"] = anual["SG_UF"] + "_" + anual["NO_MUN"].map(_norm)
    long = anual.melt(id_vars=["chave"], value_vars=year_cols, var_name="ano", value_name="fundeb_total")
    long["NU_ANO_AVALIACAO"] = long["ano"].astype(int) + 1  # lag de 1 ano
    # Mapa código IBGE -> chave (UF + nome) a partir da base de alunos
    mun = students.drop_duplicates("CO_MUNICIPIO")[["CO_MUNICIPIO", "SG_UF"]]
    nomes = _read(RAW / "alfabetizacao_inep" / "2024_municipio.csv", usecols=["CO_MUNICIPIO", "NO_MUNICIPIO"]).drop_duplicates()
    mun = mun.merge(nomes, on="CO_MUNICIPIO")
    mun["chave"] = mun["SG_UF"] + "_" + mun["NO_MUNICIPIO"].map(_norm)
    out = long.merge(mun[["CO_MUNICIPIO", "chave"]], on="chave")[["CO_MUNICIPIO", "NU_ANO_AVALIACAO", "fundeb_total"]]
    # Matrículas municipais (censo) para calcular valor por matrícula
    mat = census_df.groupby(["CO_MUNICIPIO", "NU_ANO_AVALIACAO"])["mun_mat_total"].sum().reset_index()
    out = out.merge(mat, on=["CO_MUNICIPIO", "NU_ANO_AVALIACAO"], how="left")
    out["fundeb_por_matricula"] = out["fundeb_total"] / out["mun_mat_total"].replace(0, np.nan)
    out["fundeb_total"] = np.log1p(out["fundeb_total"])  # reduz assimetria
    return out.drop(columns="mun_mat_total")


def build() -> pd.DataFrame:
    """Monta a tabela Gold de alunos (2024 e 2025: anos com lag disponível)."""
    s = load_students()
    cen = census()
    df = s.merge(school_lag(s), on=["ID_ESCOLA", "NU_ANO_AVALIACAO"], how="left")
    df = df.merge(municipality_lag(), on=["CO_MUNICIPIO", "NU_ANO_AVALIACAO", "TP_DEPENDENCIA"], how="left")
    df = df.merge(state_lag(), on=["CO_UF", "NU_ANO_AVALIACAO"], how="left")
    df = df.merge(cen.drop(columns="mun_mat_total"), on=["CO_MUNICIPIO", "TP_DEPENDENCIA", "NU_ANO_AVALIACAO"], how="left")
    df = df.merge(fundeb_lag(cen, s), on=["CO_MUNICIPIO", "NU_ANO_AVALIACAO"], how="left")
    # Região (1º dígito do código da UF) como proxy territorial agregado
    df["regiao"] = (df["CO_UF"] // 10).astype("int8")
    # 2023 não possui ano anterior -> fica fora da tabela de modelagem
    df = df[df["NU_ANO_AVALIACAO"] > min(YEARS)].reset_index(drop=True)
    # VL_PROFICIENCIA_LP só é guardada para auditoria; NUNCA é feature (define o alvo)
    return df


if __name__ == "__main__":
    GOLD.mkdir(parents=True, exist_ok=True)
    gold = build()
    gold.to_parquet(GOLD / "alunos_gold.parquet", index=False)
    print(gold.shape)
    print(gold.isna().mean().sort_values(ascending=False).head(15))
