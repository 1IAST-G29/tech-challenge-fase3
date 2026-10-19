"""Interpretabilidade: Permutation Importance e SHAP sobre o pipeline treinado."""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src.config import MODELS, REPORTS, SEED, TARGET
from src.modeling.train import load_split
from src.visualization.plots import barh, curves, save_current


def main():
    _, te, num, cat = load_split()
    model = joblib.load(MODELS / "hgb_pipeline.joblib")
    X, y = te[num + cat], te[TARGET]
    # Curvas de desempenho no teste 2025
    curves(y, model.predict_proba(X)[:, 1])

    # Permutation importance em amostra do teste (mede queda de AUC ao embaralhar cada coluna)
    s = te.sample(60000, random_state=SEED)
    pi = permutation_importance(model, s[num + cat], s[TARGET], scoring="roc_auc", n_repeats=3,
                                random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=num + cat).sort_values(ascending=False)
    imp.to_csv(REPORTS / "permutation_importance.csv", header=["queda_auc"])
    barh(imp, "Permutation importance (queda de ROC-AUC)", "importancia_permutacao.png")

    # SHAP: aplica só o pré-processamento e explica o classificador final
    pre, clf = model[:-1], model[-1]
    sh = te.sample(1500, random_state=SEED)
    Xt = pre.transform(sh[num + cat])
    names = list(pre.get_feature_names_out())
    bg = shap.sample(pre.transform(te.sample(20000, random_state=1)[num + cat]), 100, random_state=SEED)
    f = lambda a: clf.predict_proba(pd.DataFrame(a, columns=names))[:, 1]
    expl = shap.Explainer(f, pd.DataFrame(bg, columns=names), algorithm="permutation")
    sv = expl(pd.DataFrame(Xt, columns=names), max_evals=2 * len(names) + 1)
    shap.summary_plot(sv.values, pd.DataFrame(Xt, columns=names), show=False, max_display=20)
    save_current(plt.gcf(), "shap_summary.png")
    shap_imp = pd.Series(abs(sv.values).mean(0), index=names).sort_values(ascending=False)
    shap_imp.to_csv(REPORTS / "shap_importance.csv", header=["mean_abs_shap"])
    barh(shap_imp, "Importância SHAP (|valor| médio)", "shap_importancia.png")
    print(imp.head(15)); print(shap_imp.head(15))


if __name__ == "__main__":
    main()
