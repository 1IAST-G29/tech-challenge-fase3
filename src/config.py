"""Configurações centrais: caminhos, constantes e colunas usadas no projeto."""
from pathlib import Path

# Raiz do repositório (src/config.py -> sobe dois níveis)
ROOT = Path(__file__).resolve().parents[1]
# Dados brutos já versionados via Git LFS
RAW = ROOT / "src" / "data" / "notebooks"
# Camada Gold gerada localmente (ignorada pelo Git)
GOLD = ROOT / "data" / "gold"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images"

# Semente única para garantir replicabilidade
SEED = 42
# Anos de avaliação disponíveis; o primeiro só serve de "ano anterior" (lag)
YEARS = (2023, 2024, 2025)
# Treino em 2024, teste fora do tempo em 2025 (2023 não tem lag disponível)
TRAIN_YEAR, TEST_YEAR = 2024, 2025
# Alvo binário: 1 = alfabetizado
TARGET = "IN_ALFABETIZADO"
# Colunas de identificação (nunca entram como features)
ID_COLS = ["ID_ALUNO", "ID_ESCOLA", "CO_MUNICIPIO"]

# Censo Escolar: ano do censo usado = ano da avaliação - 1 (informação já disponível)
CENSO_FILES = {
    2022: RAW / "censo_escolar" / "2022_microdados_ed_basica.csv",
    2023: RAW / "censo_escolar" / "2023_microdados_ed_basica.csv",
    2024: RAW / "censo_escolar" / "2024_microdados_ed_basica.csv",
}
# Colunas do censo presentes nos três anos
CENSO_COLS = [
    "TP_LOCALIZACAO_DIFERENCIADA",
    "IN_AGUA_POTAVEL", "IN_ENERGIA_REDE_PUBLICA", "IN_ESGOTO_REDE_PUBLICA",
    "IN_BIBLIOTECA", "IN_BIBLIOTECA_SALA_LEITURA", "IN_LABORATORIO_INFORMATICA",
    "IN_INTERNET", "IN_COMPUTADOR", "IN_ALIMENTACAO", "IN_QUADRA_ESPORTES",
    "IN_PARQUE_INFANTIL", "IN_EQUIP_TV", "IN_EQUIP_LOUSA_DIGITAL",
    "QT_SALAS_UTILIZADAS", "QT_MAT_BAS", "QT_MAT_FUND_AI", "QT_DOC_FUND_AI",
    "QT_TUR_FUND_AI", "TP_LOCALIZACAO", "QT_MAT_BAS_BRANCA", "QT_MAT_BAS_PRETA", "QT_MAT_BAS_PARDA",
    "QT_MAT_BAS_INDIGENA", "QT_MAT_BAS_ND", "QT_PROF_PEDAGOGIA", "QT_PROF_COORDENADOR",
]
