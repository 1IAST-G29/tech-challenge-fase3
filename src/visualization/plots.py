"""Funções de gráficos reutilizáveis (salvam em images/)."""
import matplotlib

matplotlib.use("Agg")  # backend sem janela (roda em servidor/CI)
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from src.config import IMAGES


def _save(fig, name):
    """Salva a figura em images/ e fecha para liberar memória."""
    IMAGES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=130)
    plt.close(fig)


def curves(y, proba, name="curvas_roc_pr_calibracao.png"):
    """ROC, Precision-Recall e calibração lado a lado."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    RocCurveDisplay.from_predictions(y, proba, ax=ax[0]); ax[0].set_title("ROC (teste 2025)")
    PrecisionRecallDisplay.from_predictions(y, proba, ax=ax[1]); ax[1].set_title("Precision-Recall")
    frac, mean = calibration_curve(y, proba, n_bins=15, strategy="quantile")
    ax[2].plot(mean, frac, "o-"); ax[2].plot([0, 1], [0, 1], "--", c="gray")
    ax[2].set(title="Calibração", xlabel="Prob. prevista", ylabel="Fração observada")
    _save(fig, name)


def barh(series, title, name, top=20):
    """Barras horizontais ordenadas (importâncias, médias, etc.)."""
    s = series.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(s) + 1.5))
    ax.barh(s.index, s.values); ax.set_title(title)
    _save(fig, name)


def hist_by(df, col, by, title, name, bins=40):
    """Histogramas sobrepostos por classe."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, g in df.groupby(by):
        ax.hist(g[col].dropna(), bins=bins, alpha=0.5, density=True, label=f"{by}={k}")
    ax.set(title=title, xlabel=col); ax.legend()
    _save(fig, name)


def heatmap(corr, title, name):
    """Mapa de calor de correlações."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(im); ax.set_title(title)
    _save(fig, name)


def save_current(fig, name):
    """Salva figura externa (ex.: SHAP) em images/."""
    _save(fig, name)
