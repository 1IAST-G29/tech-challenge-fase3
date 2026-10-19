"""Executa a pipeline completa de ponta a ponta (reprodutível, sementes fixas)."""
from src.evaluation import interpret, municipal
from src.modeling import train
from src.preprocessing import build_gold
from src.visualization import eda

if __name__ == "__main__":
    build_gold.GOLD.mkdir(parents=True, exist_ok=True)
    build_gold.build().to_parquet(build_gold.GOLD / "alunos_gold.parquet", index=False)  # 1. Gold
    eda.main()  # 2. EDA
    train.main()  # 3. Treino/validação/teste
    interpret.main()  # 4. Importância + SHAP
    municipal.main()  # 5. Risco, clusters, metas
