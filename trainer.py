#!/usr/bin/env python3
"""
trainer.py  –  Full SDLC/ML-lifecycle training and evaluation pipeline
=======================================================================

Machine-Learning Lifecycle stages implemented here:
    1. Data ingestion          – loads all 10 000 tweets from NLTK corpus
    2. Data preprocessing      – process_tweet on every tweet
    3. Feature engineering     – extract_features_batch (pos/neg freq sums)
    4. Train / validation split – 80 % train, 20 % test (mirrors assignment)
    5. Model training
           a) Assignment GD (gradientDescent) – exactly matches Coursera
           b) Advanced GD   (GradientDescentTrainer) – momentum + early stop
           c) sklearn LR    (AdvancedLogisticRegression) – hyper-param tuned
    6. Evaluation              – accuracy, precision, recall, F1, ROC-AUC
    7. Error analysis          – see errors.py
    8. Persistence             – save models, frequency dict, metadata

SDLC stages covered:
    Requirements → Design → Implementation → Testing → Deployment (save)
    → Maintenance (logging, metadata, reproducibility via seed)

Usage (standalone):
    python trainer.py

Usage (API):
    from trainer import SentimentTrainer
    trainer = SentimentTrainer()
    artifacts = trainer.run()
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# project imports
import nltk
from nltk.corpus import twitter_samples

from utils import process_tweet, build_freqs
from sentiment_model import (
    sigmoid,
    gradientDescent,
    extract_features_batch,
    predict_tweet,
    evaluate_logistic_regression as test_logistic_regression,
    GradientDescentTrainer,
    AdvancedLogisticRegression,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────── #
#  Directories & constants                                            #
# ─────────────────────────────────────────────────────────────────── #
OUTPUT_DIR = Path("preprocessed_data")
MODELS_DIR = OUTPUT_DIR / "models"
VIZ_DIR    = OUTPUT_DIR / "visualizations"

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Assignment-exact hyper-parameters
GD_ALPHA      = 1e-9
GD_NUM_ITERS  = 1500
THETA_INIT    = np.zeros((3, 1))

# Advanced trainer hyper-parameters
ADV_ALPHA      = 1e-9
ADV_ITERS      = 3000
ADV_MOMENTUM   = 0.9
ADV_LAMBDA     = 1e-4
ADV_BATCH_SIZE = None    # full-batch
ADV_CHECK      = 100
ADV_PATIENCE   = 20
ADV_TOL        = 1e-10


# ─────────────────────────────────────────────────────────────────── #
#  Logger setup                                                       #
# ─────────────────────────────────────────────────────────────────── #
def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sentiment_trainer")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  [%(levelname)s]  %(message)s")
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────── #
#  Metrics helper                                                     #
# ─────────────────────────────────────────────────────────────────── #
def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """Return dict with accuracy, precision, recall, F1, ROC-AUC."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )
    y_true = np.squeeze(y_true).astype(int)
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 6),
    }


# ─────────────────────────────────────────────────────────────────── #
#  Main trainer class                                                 #
# ─────────────────────────────────────────────────────────────────── #
class SentimentTrainer:
    """End-to-end pipeline covering full ML + SDLC lifecycle.

    Attributes (populated after :meth:`run`):
        freqs:            Frequency dictionary (word, label) → count.
        theta_gd:         Weights from assignment GD.
        theta_adv:        Weights from advanced GD trainer.
        adv_lr_model:     Fitted AdvancedLogisticRegression.
        train_x / test_x: Raw tweet lists.
        train_y / test_y: Label arrays (m, 1).
        X_train / X_test: Feature matrices (m, 3).
        metrics:          Dict of metric dicts keyed by model name.
    """

    def __init__(self, output_dir: Path = OUTPUT_DIR) -> None:
        for d in (output_dir, output_dir / "models", output_dir / "visualizations"):
            d.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir
        self.logger = _setup_logger(output_dir / "models" / "trainer.log")

        # populated by run()
        self.freqs: Dict[Tuple[str, float], int] = {}
        self.theta_gd: Optional[np.ndarray] = None
        self.theta_adv: Optional[np.ndarray] = None
        self.adv_lr_model: Optional[AdvancedLogisticRegression] = None
        self.train_x: List[str] = []
        self.test_x:  List[str] = []
        self.train_y: Optional[np.ndarray] = None
        self.test_y:  Optional[np.ndarray] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test:  Optional[np.ndarray] = None
        self.metrics: Dict[str, Dict[str, float]] = {}

    # ────────────────────────────────────────────────── #
    #  Stage 1: Data ingestion                           #
    # ────────────────────────────────────────────────── #
    def _ingest(self) -> Tuple[List[str], List[str]]:
        self.logger.info("── Stage 1: Data ingestion ────────────────────────")
        for corpus in ("twitter_samples", "stopwords"):
            try:
                nltk.data.find(f"corpora/{corpus}")
            except LookupError:
                self.logger.info(f"Downloading NLTK corpus: {corpus}")
                nltk.download(corpus, quiet=True)

        pos_tweets = twitter_samples.strings("positive_tweets.json")  # 5 000
        neg_tweets = twitter_samples.strings("negative_tweets.json")  # 5 000

        self.logger.info(f"Loaded {len(pos_tweets)} positive + {len(neg_tweets)} negative tweets")
        self.logger.info(f"Total corpus: {len(pos_tweets) + len(neg_tweets)} tweets")
        return pos_tweets, neg_tweets

    # ────────────────────────────────────────────────── #
    #  Stage 2: Split (assignment-exact)                 #
    # ────────────────────────────────────────────────── #
    def _split(
        self, pos_tweets: List[str], neg_tweets: List[str]
    ) -> None:
        self.logger.info("── Stage 2: Train / test split ────────────────────")
        # Assignment convention: first 4000 of each → train, last 1000 → test
        train_pos = pos_tweets[:4000]
        train_neg = neg_tweets[:4000]
        test_pos  = pos_tweets[4000:]   # 1 000
        test_neg  = neg_tweets[4000:]   # 1 000

        self.train_x = train_pos + train_neg   # 8 000
        self.test_x  = test_pos  + test_neg    # 2 000

        self.train_y = np.append(
            np.ones((len(train_pos), 1)),
            np.zeros((len(train_neg), 1)), axis=0
        )                                      # (8000, 1)
        self.test_y = np.append(
            np.ones((len(test_pos), 1)),
            np.zeros((len(test_neg), 1)), axis=0
        )                                      # (2000, 1)

        self.logger.info(
            f"train_x: {len(self.train_x)} tweets  "
            f"(pos={len(train_pos)}, neg={len(train_neg)})"
        )
        self.logger.info(
            f"test_x:  {len(self.test_x)} tweets  "
            f"(pos={len(test_pos)}, neg={len(test_neg)})"
        )
        self.logger.info(f"train_y shape: {self.train_y.shape}")
        self.logger.info(f"test_y  shape: {self.test_y.shape}")

    # ────────────────────────────────────────────────── #
    #  Stage 3: Preprocessing + frequency dict           #
    # ────────────────────────────────────────────────── #
    def _preprocess(self) -> None:
        self.logger.info("── Stage 3: Preprocessing + freq dict ─────────────")
        t0 = time.perf_counter()

        # Verify all tweets are processed
        self.logger.info("Processing all training tweets to build freq dict…")
        self.freqs = build_freqs(self.train_x, self.train_y)

        elapsed = time.perf_counter() - t0
        self.logger.info(
            f"Freq dict built in {elapsed:.2f}s  |  unique (word,label) pairs: {len(self.freqs)}"
        )
        # sanity-check example from assignment
        self._log_process_tweet_example()

    def _log_process_tweet_example(self) -> None:
        """Log the exact expected outputs from the assignment."""
        example = self.train_x[0]
        processed = process_tweet(example)
        self.logger.info(f"process_tweet example:")
        self.logger.info(f"  raw:       {example}")
        self.logger.info(f"  processed: {processed}")

    # ────────────────────────────────────────────────── #
    #  Stage 4: Feature engineering                      #
    # ────────────────────────────────────────────────── #
    def _featurise(self) -> None:
        self.logger.info("── Stage 4: Feature engineering ───────────────────")
        t0 = time.perf_counter()

        self.logger.info(f"Building X_train from {len(self.train_x)} tweets…")
        self.X_train = extract_features_batch(self.train_x, self.freqs)  # (8000, 3)

        self.logger.info(f"Building X_test  from {len(self.test_x)} tweets…")
        self.X_test  = extract_features_batch(self.test_x,  self.freqs)  # (2000, 3)

        elapsed = time.perf_counter() - t0
        self.logger.info(f"Feature extraction done in {elapsed:.2f}s")
        self.logger.info(f"X_train shape: {self.X_train.shape}")
        self.logger.info(f"X_test  shape: {self.X_test.shape}")

        # feature distribution summary
        for col_idx, name in enumerate(["bias", "positive_sum", "negative_sum"]):
            col = self.X_train[:, col_idx]
            self.logger.info(
                f"  {name:15s}  min={col.min():.1f}  max={col.max():.1f}  "
                f"mean={col.mean():.1f}  std={col.std():.1f}"
            )

    # ────────────────────────────────────────────────── #
    #  Stage 5a: Assignment gradient descent             #
    # ────────────────────────────────────────────────── #
    def _train_assignment_gd(self) -> None:
        self.logger.info("── Stage 5a: Assignment gradientDescent ────────────")
        self.logger.info(
            f"alpha={GD_ALPHA}, num_iters={GD_NUM_ITERS}, theta_init=zeros(3,1)"
        )
        t0 = time.perf_counter()
        J, theta = gradientDescent(
            self.X_train, self.train_y,
            THETA_INIT.copy(), GD_ALPHA, GD_NUM_ITERS
        )
        elapsed = time.perf_counter() - t0
        self.theta_gd = theta

        self.logger.info(f"Training done in {elapsed:.2f}s")
        self.logger.info(f"Final cost J = {J:.8f}")
        self.logger.info(
            f"Theta = {[round(float(t), 8) for t in np.squeeze(theta)]}"
        )

        # expected output from assignment:
        # J ≈ 0.24216529,  theta ≈ [7e-08, 0.0005239, -0.00055517]
        self._eval_model("assignment_gd", theta, sigmoid_theta=True)
        self._log_sample_predictions(theta, sigmoid_theta=True)

    # ────────────────────────────────────────────────── #
    #  Stage 5b: Advanced GD (momentum + early stopping) #
    # ────────────────────────────────────────────────── #
    def _train_advanced_gd(self) -> None:
        self.logger.info("── Stage 5b: Advanced GradientDescentTrainer ───────")
        self.logger.info(
            f"alpha={ADV_ALPHA}, iters={ADV_ITERS}, momentum={ADV_MOMENTUM}, "
            f"lambda={ADV_LAMBDA}, patience={ADV_PATIENCE}"
        )
        t0 = time.perf_counter()
        trainer = GradientDescentTrainer(
            alpha=ADV_ALPHA,
            num_iters=ADV_ITERS,
            momentum=ADV_MOMENTUM,
            lambda_=ADV_LAMBDA,
            tol=ADV_TOL,
            patience=ADV_PATIENCE,
            check_every=ADV_CHECK,
            batch_size=ADV_BATCH_SIZE,
            verbose=ADV_CHECK,
        )
        trainer.fit(self.X_train, self.train_y, theta_init=THETA_INIT.copy())
        elapsed = time.perf_counter() - t0

        self.theta_adv = trainer.theta_
        hist = trainer.history_
        self.logger.info(
            f"Training done in {elapsed:.2f}s  |  "
            f"iterations={hist.n_iters}  converged={hist.converged}"
        )
        if hist.costs:
            self.logger.info(
                f"Cost: start={hist.costs[0]:.6f} → end={hist.costs[-1]:.6f}"
            )

        # save cost history for plotting
        cost_path = self.output_dir / "models" / "advanced_gd_costs.json"
        with open(cost_path, "w") as f:
            json.dump(hist.costs, f)

        self._eval_model("advanced_gd", trainer.theta_, prob_fn=trainer.predict_proba)

    # ────────────────────────────────────────────────── #
    #  Stage 5c: sklearn with hyper-param grid search    #
    # ────────────────────────────────────────────────── #
    def _train_sklearn(self) -> None:
        self.logger.info("── Stage 5c: AdvancedLogisticRegression (sklearn) ──")
        t0 = time.perf_counter()
        model = AdvancedLogisticRegression(
            cv_folds=5, max_iter=1000, random_state=RANDOM_SEED, verbose=False
        )
        model.fit(self.X_train, self.train_y)
        elapsed = time.perf_counter() - t0

        self.adv_lr_model = model
        self.logger.info(f"sklearn grid-search done in {elapsed:.2f}s")
        self.logger.info(f"Best params: {model.best_params_}")

        y_pred  = model.predict(self.X_test)
        y_prob  = model.predict_proba(self.X_test)
        metrics = _compute_metrics(self.test_y, y_pred, y_prob)
        self.metrics["sklearn_tuned"] = metrics
        self._log_metrics("sklearn_tuned", metrics)

    # ────────────────────────────────────────────────── #
    #  Evaluation helper                                 #
    # ────────────────────────────────────────────────── #
    def _eval_model(
        self,
        name: str,
        theta: np.ndarray,
        sigmoid_theta: bool = False,
        prob_fn=None,
    ) -> None:
        """Compute and log full metrics on the test set."""
        # accuracy via assignment function
        acc = test_logistic_regression(self.test_x, self.test_y, self.freqs, theta)
        self.logger.info(f"{name}  accuracy via test_logistic_regression = {acc:.4f}")

        # full metrics via sklearn
        if prob_fn is not None:
            y_prob = prob_fn(self.X_test)
        else:
            y_prob = sigmoid(self.X_test @ theta).ravel()

        y_pred = (y_prob >= 0.5).astype(int)
        metrics = _compute_metrics(self.test_y, y_pred, y_prob)
        self.metrics[name] = metrics
        self._log_metrics(name, metrics)

    def _log_metrics(self, name: str, m: Dict[str, float]) -> None:
        self.logger.info(
            f"  {name}  acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
            f"rec={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}"
        )

    def _log_sample_predictions(self, theta: np.ndarray, sigmoid_theta: bool = True) -> None:
        """Replicate the 'sample tweets' block from the Coursera notebook."""
        sample_tweets = [
            "I am happy",
            "I am bad",
            "this movie should have been great.",
            "great",
            "great great",
            "great great great",
            "great great great great",
        ]
        self.logger.info("Sample predictions (assignment section 4):")
        for tw in sample_tweets:
            prob = float(predict_tweet(tw, self.freqs, theta))
            self.logger.info(f"  {tw!r:45s} → {prob:.6f}")

    # ────────────────────────────────────────────────── #
    #  Stage 6: Visualisations                           #
    # ────────────────────────────────────────────────── #
    def _visualise(self) -> None:
        self.logger.info("── Stage 6: Visualisations ─────────────────────────")
        try:
            from visualiser import Visualiser
            viz = Visualiser(
                X_test=self.X_test,
                y_test=self.test_y,
                theta_gd=self.theta_gd,
                theta_adv=self.theta_adv,
                sklearn_model=self.adv_lr_model,
                freq_dict=self.freqs,
                cost_history_path=self.output_dir / "models" / "advanced_gd_costs.json",
                output_dir=self.output_dir / "visualizations",
            )
            viz.plot_all()
        except Exception as exc:
            self.logger.warning(f"Visualisation skipped: {exc}")

    # ────────────────────────────────────────────────── #
    #  Stage 7: Persistence                              #
    # ────────────────────────────────────────────────── #
    def _persist(self) -> None:
        self.logger.info("── Stage 7: Saving artefacts ───────────────────────")
        md = self.output_dir / "models"
        md.mkdir(exist_ok=True)

        # freq dict
        with open(md / "freqs.pkl", "wb") as f:
            pickle.dump(self.freqs, f)

        # assignment theta
        if self.theta_gd is not None:
            np.save(md / "theta_gd.npy", self.theta_gd)

        # advanced GD theta
        if self.theta_adv is not None:
            np.save(md / "theta_adv.npy", self.theta_adv)

        # sklearn model
        if self.adv_lr_model is not None:
            with open(md / "sklearn_tuned_model.pkl", "wb") as f:
                pickle.dump(self.adv_lr_model, f)

        # metadata / results JSON
        meta = {
            "timestamp": datetime.now().isoformat(),
            "random_seed": RANDOM_SEED,
            "train_size": len(self.train_x),
            "test_size":  len(self.test_x),
            "freq_dict_size": len(self.freqs),
            "gd_hyparams": {
                "alpha": GD_ALPHA, "num_iters": GD_NUM_ITERS,
            },
            "advanced_gd_hyparams": {
                "alpha": ADV_ALPHA, "num_iters": ADV_ITERS,
                "momentum": ADV_MOMENTUM, "lambda": ADV_LAMBDA,
            },
            "sklearn_best_params": (
                self.adv_lr_model.best_params_ if self.adv_lr_model else {}
            ),
            "metrics": self.metrics,
        }
        with open(md / "training_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.logger.info(f"Saved artefacts to {md}")

    # ────────────────────────────────────────────────── #
    #  Public: run complete pipeline                     #
    # ────────────────────────────────────────────────── #
    def run(self) -> Dict[str, Any]:
        """Execute all pipeline stages in order.

        Returns:
            Dict with keys: freqs, theta_gd, theta_adv, metrics, …
        """
        self.logger.info("=" * 70)
        self.logger.info(" SENTIMENT ANALYSIS – FULL PIPELINE")
        self.logger.info(f" Started: {datetime.now().isoformat()}")
        self.logger.info("=" * 70)

        t_start = time.perf_counter()

        pos_tweets, neg_tweets = self._ingest()
        self._split(pos_tweets, neg_tweets)
        self._preprocess()
        self._featurise()
        self._train_assignment_gd()
        self._train_advanced_gd()
        self._train_sklearn()
        self._visualise()
        self._persist()

        elapsed = time.perf_counter() - t_start
        self.logger.info("=" * 70)
        self.logger.info(f" PIPELINE COMPLETE  ({elapsed:.1f}s)")
        self.logger.info("=" * 70)

        return {
            "freqs":          self.freqs,
            "theta_gd":       self.theta_gd,
            "theta_adv":      self.theta_adv,
            "sklearn_model":  self.adv_lr_model,
            "metrics":        self.metrics,
            "train_x":        self.train_x,
            "test_x":         self.test_x,
            "train_y":        self.train_y,
            "test_y":         self.test_y,
            "X_train":        self.X_train,
            "X_test":         self.X_test,
        }


# ─────────────────────────────────────────────────────────────────── #
#  CLI entry-point                                                    #
# ─────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    trainer = SentimentTrainer()
    artefacts = trainer.run()
