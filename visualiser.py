#!/usr/bin/env python3
"""
visualiser.py  –  All plots for the sentiment analysis pipeline
===============================================================

Plots produced:
    1. cost_history.png          – training loss curves (GD vs advanced GD)
    2. decision_boundary_gd.png  – pos/neg freq scatter + GD boundary
    3. roc_curves.png            – ROC curves for all three models
    4. confusion_matrices.png    – side-by-side confusion matrices
    5. feature_distribution.png  – box plots of pos/neg freq sums by class

Usage (standalone):
    python visualiser.py

Usage (API):
    from visualiser import Visualiser
    viz = Visualiser(...)
    viz.plot_all()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────── #
#  Lazy matplotlib import guard                                       #
# ─────────────────────────────────────────────────────────────────── #
def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend (safe for CI)
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for visualisations: pip install matplotlib") from exc


class Visualiser:
    """Produces and saves all plots for the sentiment pipeline.

    Parameters
    ----------
    X_test, y_test      : test feature matrix (m,3) and labels (m,1)
    theta_gd            : assignment GD weights (3,1)
    theta_adv           : advanced GD weights (3,1), optional
    sklearn_model       : fitted AdvancedLogisticRegression, optional
    freq_dict           : build_freqs output
    cost_history_path   : JSON file produced by trainer (advanced GD costs)
    output_dir          : directory to write PNGs into
    """

    def __init__(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        theta_gd: np.ndarray,
        theta_adv: Optional[np.ndarray] = None,
        sklearn_model=None,
        freq_dict: Optional[Dict] = None,
        cost_history_path: Optional[Path] = None,
        output_dir: Path = Path("preprocessed_data/visualizations"),
    ) -> None:
        self.X_test   = X_test
        self.y_test   = np.squeeze(y_test)
        self.theta_gd = theta_gd
        self.theta_adv = theta_adv
        self.sklearn_model = sklearn_model
        self.freq_dict = freq_dict
        self.cost_path = Path(cost_history_path) if cost_history_path else None
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────── #
    def plot_all(self) -> None:
        """Render + save every plot."""
        fns = [
            self._plot_cost_history,
            self._plot_decision_boundary,
            self._plot_roc_curves,
            self._plot_confusion_matrices,
            self._plot_feature_distribution,
        ]
        for fn in fns:
            try:
                fn()
            except Exception as exc:
                logger.warning(f"{fn.__name__} failed: {exc}")

    # ────────────────────────────────────────────────── #
    def _plot_cost_history(self) -> None:
        if self.cost_path is None or not self.cost_path.exists():
            return
        plt = _get_plt()
        with open(self.cost_path) as f:
            costs = json.load(f)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(costs, linewidth=1.5, color="steelblue")
        ax.set_xlabel("Checkpoint (every 100 iterations)")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title("Advanced GD – Training cost history")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.out / "cost_history_advanced_gd.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {out}")

    # ────────────────────────────────────────────────── #
    def _plot_decision_boundary(self) -> None:
        """Scatter pos- vs neg-freq sums; overlay the GD decision boundary."""
        plt = _get_plt()
        x_feat = np.log1p(self.X_test[:, 1])   # log1p for readability
        y_feat = np.log1p(self.X_test[:, 2])
        labels = self.y_test.astype(int)

        fig, ax = plt.subplots(figsize=(7, 6))
        for cls, colour, marker, label in [
            (1, "#2196F3", "o", "Positive"),
            (0, "#F44336", "x", "Negative"),
        ]:
            mask = labels == cls
            ax.scatter(x_feat[mask], y_feat[mask], c=colour, marker=marker,
                       alpha=0.4, s=20, label=label)

        # decision boundary  θ₀ + θ₁·x + θ₂·y = 0  →  y = -(θ₀ + θ₁·x) / θ₂
        th = np.squeeze(self.theta_gd)
        x_line = np.linspace(x_feat.min(), x_feat.max(), 200)
        if abs(th[2]) > 1e-15:
            y_line = -(th[0] + th[1] * x_line) / th[2]
            ax.plot(x_line, y_line, "k--", linewidth=1.5, label="GD boundary")

        ax.set_xlabel("log(1+pos_freq_sum)")
        ax.set_ylabel("log(1+neg_freq_sum)")
        ax.set_title("Decision boundary – assignment GD")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.out / "decision_boundary_gd.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {out}")

    # ────────────────────────────────────────────────── #
    def _plot_roc_curves(self) -> None:
        from sklearn.metrics import roc_curve, auc
        from sentiment_model import sigmoid
        plt = _get_plt()

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k:", linewidth=0.8)

        models = [("Assignment GD", self.theta_gd, "#2196F3")]
        if self.theta_adv is not None:
            models.append(("Advanced GD", self.theta_adv, "#4CAF50"))

        for model_name, theta, colour in models:
            y_prob = sigmoid(self.X_test @ theta).ravel()
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colour, linewidth=2,
                    label=f"{model_name}  (AUC={roc_auc:.4f})")

        if self.sklearn_model is not None:
            y_prob_sk = self.sklearn_model.predict_proba(self.X_test)
            fpr, tpr, _ = roc_curve(self.y_test, y_prob_sk)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color="#FF9800", linewidth=2,
                    label=f"sklearn tuned  (AUC={roc_auc:.4f})")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves – all models")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.out / "roc_curves_all_models.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {out}")

    # ────────────────────────────────────────────────── #
    def _plot_confusion_matrices(self) -> None:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        from sentiment_model import sigmoid
        plt = _get_plt()

        model_data = [("Assignment GD", sigmoid(self.X_test @ self.theta_gd).ravel())]
        if self.theta_adv is not None:
            model_data.append(("Advanced GD", sigmoid(self.X_test @ self.theta_adv).ravel()))
        if self.sklearn_model is not None:
            model_data.append(("sklearn tuned", self.sklearn_model.predict_proba(self.X_test)))

        n = len(model_data)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (name, probs) in zip(axes, model_data):
            preds = (probs >= 0.5).astype(int)
            cm = confusion_matrix(self.y_test.astype(int), preds)
            disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(name)

        plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = self.out / "confusion_matrices.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {out}")

    # ────────────────────────────────────────────────── #
    def _plot_feature_distribution(self) -> None:
        plt = _get_plt()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        labels_int = self.y_test.astype(int)

        for col_idx, (feat_name, ax) in enumerate(
            zip(["pos_freq_sum", "neg_freq_sum"], axes)
        ):
            pos_vals = np.log1p(self.X_test[labels_int == 1, col_idx + 1])
            neg_vals = np.log1p(self.X_test[labels_int == 0, col_idx + 1])
            ax.boxplot([pos_vals, neg_vals], labels=["Positive tweets", "Negative tweets"],
                       patch_artist=True,
                       boxprops=dict(facecolor="#90CAF9" if col_idx == 0 else "#EF9A9A"))
            ax.set_ylabel(f"log(1 + {feat_name})")
            ax.set_title(f"Distribution of {feat_name} by class")
            ax.grid(True, alpha=0.3)

        plt.suptitle("Feature Distributions – test set", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = self.out / "feature_distribution.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────── #
#  Standalone CLI                                                     #
# ─────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    import pickle
    MODELS = Path("preprocessed_data/models")

    with open(MODELS / "freqs.pkl", "rb") as f:
        freqs = pickle.load(f)
    theta_gd  = np.load(MODELS / "theta_gd.npy")
    theta_adv = np.load(MODELS / "theta_adv.npy") if (MODELS / "theta_adv.npy").exists() else None

    try:
        with open(MODELS / "sklearn_tuned_model.pkl", "rb") as f:
            sklearn_model = pickle.load(f)
    except FileNotFoundError:
        sklearn_model = None

    import nltk
    nltk.download("twitter_samples", quiet=True)
    from nltk.corpus import twitter_samples
    from sentiment_model import extract_features_batch
    pos = twitter_samples.strings("positive_tweets.json")
    neg = twitter_samples.strings("negative_tweets.json")
    test_x = pos[4000:] + neg[4000:]
    test_y = np.append(np.ones((1000, 1)), np.zeros((1000, 1)), axis=0)
    X_test = extract_features_batch(test_x, freqs)

    viz = Visualiser(
        X_test=X_test,
        y_test=test_y,
        theta_gd=theta_gd,
        theta_adv=theta_adv,
        sklearn_model=sklearn_model,
        freq_dict=freqs,
        cost_history_path=MODELS / "advanced_gd_costs.json",
    )
    viz.plot_all()
