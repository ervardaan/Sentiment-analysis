#!/usr/bin/env python3
"""
sentiment_model.py  –  Core logistic-regression functions
==========================================================

Implements every function from the Coursera NLP Specialisation Week-1
assignment, plus production-grade extensions:

    sigmoid(z)
    cost_function(theta, x, y)          ← vectorised, clipped
    gradientDescent(x, y, theta, alpha, num_iters)
    extract_features(tweet, freqs)
    predict_tweet(tweet, freqs, theta)
    evaluate_logistic_regression(test_x, test_y, freqs, theta)

Advanced additions:
    GradientDescentTrainer              ← full-history, early-stopping, momentum
    AdvancedLogisticRegression          ← sklearn wrapper with hyper-param tuning
    extract_features_batch(tweets, freqs) ← vectorised batch extraction
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from utils import process_tweet

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Part 1.1  –  Sigmoid                                                       #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """σ(z) = 1 / (1 + e^{-z})

    Numerically stable: clamps the exponent to avoid overflow for very
    large negative z values.  Works on scalars, 1-D and 2-D arrays.

    Args:
        z: Logit value(s) – scalar, list, or any numpy array.

    Returns:
        Sigmoid-transformed value(s) in the open interval (0, 1).

    Examples:
        >>> sigmoid(0)
        0.5
        >>> sigmoid(4.92)
        0.9927537604041685
        >>> sigmoid(np.array([-1, 0, 1]))
        array([0.26894142, 0.5       , 0.73105858])
    """
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Part 1.2  –  Cost function                                                 #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def cost_function(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Binary cross-entropy cost  J(θ).

    J(θ) = -1/m  Σ [ y·log(h) + (1-y)·log(1-h) ]
    where  h = σ(x·θ).

    Args:
        theta: Weight vector (n+1, 1) or (n+1,).
        x:     Feature matrix (m, n+1).
        y:     Label vector   (m, 1)  or (m,).

    Returns:
        Scalar cost value.
    """
    m = x.shape[0]
    h = sigmoid(x @ theta)
    h = np.clip(h, 1e-15, 1 - 1e-15)
    J = -1.0 / m * (
        np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))
    )
    return float(J)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Part 1.2  –  Gradient descent (exact Coursera API)                         #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def gradientDescent(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    alpha: float,
    num_iters: int,
) -> Tuple[float, np.ndarray]:
    """Train logistic regression via full-batch gradient descent.

    Update rule (vectorised):
        h     = σ(x · θ)
        J     = -1/m · [ yᵀ·log(h) + (1-y)ᵀ·log(1-h) ]
        θ  ←  θ - (α/m) · xᵀ·(h - y)

    Args:
        x:         Feature matrix  (m, n+1)  with bias column pre-pended.
        y:         Label vector    (m, 1).
        theta:     Initial weights (n+1, 1).
        alpha:     Learning rate   (e.g. 1e-9).
        num_iters: Number of full-dataset passes.

    Returns:
        J:     Final scalar cost.
        theta: Trained weight vector (n+1, 1).

    Examples:
        >>> J, theta = gradientDescent(X, Y, np.zeros((3,1)), 1e-9, 1500)
        >>> # J ≈ 0.24216529,  theta ≈ [7e-08, 5.239e-04, -5.5517e-04]
    """
    m = x.shape[0]

    for _ in range(num_iters):
        z = np.dot(x, theta)            # (m, 1)
        h = sigmoid(z)                  # (m, 1)
        h = np.clip(h, 1e-15, 1 - 1e-15)

        # Vectorised cost
        J = -1.0 / m * (
            np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))
        )

        # Vectorised gradient step
        theta = theta - (alpha / m) * np.dot(x.T, (h - y))

    return float(np.squeeze(J)), theta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Part 2  –  Feature extraction                                              #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def extract_features(
    tweet: str,
    freqs: Dict[Tuple[str, float], int],
) -> np.ndarray:
    """Map a single tweet to a (1, 3) feature vector.

    Feature layout:
        x[0, 0] = 1           (bias)
        x[0, 1] = Σ freqs[(w, 1.0)]   (positive-word-frequency sum)
        x[0, 2] = Σ freqs[(w, 0.0)]   (negative-word-frequency sum)

    Args:
        tweet: Raw tweet string.
        freqs: Frequency dictionary from ``build_freqs``.

    Returns:
        Feature vector of shape (1, 3).

    Examples:
        >>> extract_features(train_x[0], freqs)
        array([[1.00e+00, 3.02e+03, 6.10e+01]])
        >>> extract_features('blorb bleeeeb bloooob', freqs)
        array([[1., 0., 0.]])
    """
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3), dtype=float)
    x[0, 0] = 1.0  # bias

    for word in word_l:
        x[0, 1] += freqs.get((word, 1.0), 0)
        x[0, 2] += freqs.get((word, 0.0), 0)

    assert x.shape == (1, 3), f"extract_features: expected (1,3), got {x.shape}"
    return x


def extract_features_batch(
    tweets: List[str],
    freqs: Dict[Tuple[str, float], int],
) -> np.ndarray:
    """Map a list of tweets to an (m, 3) feature matrix.

    Equivalent to stacking ``extract_features`` per tweet, but more
    memory-efficient for large corpora.

    Args:
        tweets: List of m raw tweet strings.
        freqs:  Frequency dictionary from ``build_freqs``.

    Returns:
        Feature matrix of shape (m, 3).
    """
    m = len(tweets)
    X = np.zeros((m, 3), dtype=float)
    X[:, 0] = 1.0  # bias column

    for i, tweet in enumerate(tweets):
        for word in process_tweet(tweet):
            X[i, 1] += freqs.get((word, 1.0), 0)
            X[i, 2] += freqs.get((word, 0.0), 0)

    return X


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Part 4  –  Prediction                                                      #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def predict_tweet(
    tweet: str,
    freqs: Dict[Tuple[str, float], int],
    theta: np.ndarray,
) -> np.ndarray:
    """Predict the positive-sentiment probability for one tweet.

    ŷ = σ(x · θ)

    Args:
        tweet: Raw tweet string.
        freqs: Frequency dictionary from ``build_freqs``.
        theta: Trained weight vector (3, 1).

    Returns:
        Probability array of shape (1, 1), value in (0, 1).
        >0.5 → positive sentiment; ≤0.5 → negative sentiment.

    Example:
        >>> predict_tweet('I am happy', freqs, theta)
        array([[0.51858...]])
    """
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Part 4  –  Accuracy on test set                                            #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def evaluate_logistic_regression(
    test_x: List[str],
    test_y: np.ndarray,
    freqs: Dict[Tuple[str, float], int],
    theta: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Evaluate accuracy of the trained model on a held-out test set.

    Args:
        test_x:     List of m raw tweet strings.
        test_y:     Label array (m, 1) with values 0.0 or 1.0.
        freqs:      Frequency dictionary from ``build_freqs``.
        theta:      Trained weights (3, 1).
        threshold:  Decision boundary (default 0.5).

    Returns:
        Accuracy ∈ [0, 1].

    Example:
        >>> evaluate_logistic_regression(test_x, test_y, freqs, theta)
        0.995
    """
    y_hat: List[int] = []

    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)
        y_hat.append(1 if float(np.squeeze(y_pred)) > threshold else 0)

    y_hat_arr = np.array(y_hat)
    y_true_arr = np.squeeze(test_y).astype(int)

    accuracy = float(np.mean(y_hat_arr == y_true_arr))
    return accuracy





# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Advanced: full-history trainer with early stopping + momentum              #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

class TrainingHistory(NamedTuple):
    costs: List[float]
    theta: np.ndarray
    converged: bool
    n_iters: int


class GradientDescentTrainer:
    """Advanced gradient-descent trainer for logistic regression.

    Features beyond the basic Coursera implementation:
    - Records full cost history for plotting
    - Momentum (Polyak heavy-ball) to accelerate convergence
    - Early stopping: halts when improvement < ``tol`` for
      ``patience`` consecutive checks
    - L2 regularisation (optional)
    - Mini-batch or full-batch mode

    Args:
        alpha:       Base learning rate.
        num_iters:   Maximum number of iterations.
        momentum:    Momentum coefficient β ∈ [0, 1)  (0 = vanilla GD).
        lambda_:     L2 regularisation strength (0 = no regularisation).
        tol:         Early-stopping tolerance.
        patience:    Early-stopping patience (checks every ``check_every``).
        check_every: Interval at which to evaluate cost for early stopping.
        batch_size:  Mini-batch size (None = full-batch).
        verbose:     Print progress every ``verbose`` iters (0 = silent).
    """

    def __init__(
        self,
        alpha: float = 1e-9,
        num_iters: int = 1500,
        momentum: float = 0.0,
        lambda_: float = 0.0,
        tol: float = 1e-8,
        patience: int = 10,
        check_every: int = 100,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        self.alpha = alpha
        self.num_iters = num_iters
        self.momentum = momentum
        self.lambda_ = lambda_
        self.tol = tol
        self.patience = patience
        self.check_every = check_every
        self.batch_size = batch_size
        self.verbose = verbose

        self.theta_: Optional[np.ndarray] = None
        self.history_: Optional[TrainingHistory] = None

    # ------------------------------------------------------------------ #

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        theta_init: Optional[np.ndarray] = None,
    ) -> "GradientDescentTrainer":
        """Train the model.

        Args:
            x:          Feature matrix (m, n+1).
            y:          Labels (m, 1).
            theta_init: Initial weights; zeros if None.

        Returns:
            self  (for method chaining).
        """
        m, n = x.shape
        theta = (
            np.zeros((n, 1)) if theta_init is None
            else theta_init.copy().reshape(n, 1)
        )
        velocity = np.zeros_like(theta)  # for momentum
        costs: List[float] = []
        stagnant = 0
        converged = False
        best_cost = float("inf")

        for it in range(self.num_iters):

            # ── mini-batch or full-batch ──────────────────────────────── #
            if self.batch_size is not None and self.batch_size < m:
                idx = np.random.choice(m, self.batch_size, replace=False)
                xb, yb = x[idx], y[idx]
                mb = self.batch_size
            else:
                xb, yb, mb = x, y, m

            # ── forward pass ─────────────────────────────────────────── #
            h = sigmoid(xb @ theta)
            h = np.clip(h, 1e-15, 1 - 1e-15)

            # ── gradient ─────────────────────────────────────────────── #
            grad = xb.T @ (h - yb) / mb

            if self.lambda_ > 0:
                reg = (self.lambda_ / mb) * theta
                reg[0] = 0.0          # don't regularise bias
                grad = grad + reg

            # ── momentum update ──────────────────────────────────────── #
            velocity = self.momentum * velocity + self.alpha * grad
            theta = theta - velocity

            # ── cost + early-stopping check ──────────────────────────── #
            if (it + 1) % self.check_every == 0:
                h_full = sigmoid(x @ theta)
                h_full = np.clip(h_full, 1e-15, 1 - 1e-15)
                J = float(np.squeeze(
                    -1.0 / m * (
                        y.T @ np.log(h_full)
                        + (1 - y).T @ np.log(1 - h_full)
                    )
                ))
                costs.append(J)

                if self.verbose and (it + 1) % self.verbose == 0:
                    logger.info(f"  iter {it+1:6d}  cost = {J:.8f}")

                if best_cost - J < self.tol:
                    stagnant += 1
                    if stagnant >= self.patience:
                        converged = True
                        if self.verbose:
                            logger.info(f"  Early stop at iter {it+1}: Δcost < {self.tol}")
                        break
                else:
                    stagnant = 0
                    best_cost = J

        self.theta_ = theta
        self.history_ = TrainingHistory(costs, theta.copy(), converged, it + 1)
        return self

    # ------------------------------------------------------------------ #

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities, shape (m,)."""
        if self.theta_ is None:
            raise RuntimeError("Call fit() first.")
        return sigmoid(x @ self.theta_).ravel()

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions, shape (m,)."""
        return (self.predict_proba(x) >= threshold).astype(int)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on (x, y)."""
        y_pred = self.predict(x)
        return float(np.mean(y_pred == np.squeeze(y).astype(int)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Advanced: sklearn wrapper with hyper-parameter search                      #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

try:
    from sklearn.linear_model import LogisticRegression as _SkLR
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class AdvancedLogisticRegression:
    """Production-grade LR backed by scikit-learn with grid-search tuning.

    Wraps ``sklearn.linear_model.LogisticRegression`` and performs an
    exhaustive grid-search over a small hyper-parameter space to find the
    best regularisation strength (C) and solver.

    Falls back gracefully when sklearn is unavailable.
    """

    # tuning grid
    PARAM_GRID = {
        "C":       [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver":  ["lbfgs", "liblinear", "saga"],
        "penalty": ["l2"],
    }

    def __init__(
        self,
        cv_folds: int = 5,
        max_iter: int = 1000,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for AdvancedLogisticRegression.")
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.best_model_: Optional[_SkLR] = None
        self.best_params_: dict = {}
        self.cv_results_: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdvancedLogisticRegression":
        """Run grid-search CV and retain the best estimator.

        Args:
            X: Feature matrix (m, n).
            y: Label vector   (m,) or (m, 1).

        Returns:
            self
        """
        y_flat = np.squeeze(y).astype(int)

        base = _SkLR(max_iter=self.max_iter, random_state=self.random_state)
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        gs = GridSearchCV(
            base,
            self.PARAM_GRID,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
            verbose=1 if self.verbose else 0,
        )
        gs.fit(X, y_flat)

        self.best_model_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.cv_results_ = {
            "mean_test_score": gs.cv_results_["mean_test_score"].tolist(),
            "std_test_score":  gs.cv_results_["std_test_score"].tolist(),
            "params":          gs.cv_results_["params"],
        }
        logger.info(f"Best params: {self.best_params_}")
        logger.info(f"Best CV AUC: {gs.best_score_:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities (m,)."""
        if self.best_model_ is None:
            raise RuntimeError("Call fit() first.")
        return self.best_model_.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions (m,)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy."""
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.squeeze(y).astype(int)))
