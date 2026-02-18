#!/usr/bin/env python3
"""
Robust Logistic Regression Model with Advanced Visualization
=============================================================

Implements state-of-the-art logistic regression modeling for sentiment analysis
with comprehensive visualizations, model evaluation, and diagnostic tools.

Features:
- Advanced feature extraction (positive/negative word sums with normalization)
- Multiple logistic regression implementations (scikit-learn + custom)
- K-fold cross-validation with stratification
- Decision boundary visualization with prediction confidence
- ROC-AUC curves and confusion matrices
- Model persistence and serialization
- Comprehensive logging and error handling

Usage:
    python logistic_regression_model.py
"""

from __future__ import annotations

import os
import json
import pickle
import logging
import warnings
from typing import Tuple, Dict, List, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.special import expit as sigmoid
from scipy.optimize import minimize

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from nltk.corpus import twitter_samples
from tqdm import tqdm

from tweet_preprocessing import TweetPreprocessor, build_freqs

warnings.filterwarnings('ignore')


# Configuration
OUTPUT_DIR = "preprocessed_data"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
LOGISTIC_FEATURES_FILE = os.path.join(OUTPUT_DIR, "logistic_features.csv")

# Split parameters
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42
CV_FOLDS = 5

# Feature extraction parameters
MIN_FEATURE_FREQ = 1
USE_NORMALIZED_FEATURES = True
USE_BIAS_TERM = True

# Model parameters
SOLVER = 'lbfgs'  # ['lbfgs', 'liblinear', 'saga']
MAX_ITER = 1000
PENALTY = 'l2'  # ['l1', 'l2', 'elasticnet']
C = 1.0  # Inverse of regularization strength
RANDOM_SEED = 42


class CustomLogisticRegression:
    """Custom logistic regression implementation using gradient descent.
    
    Implements logistic regression with sigmoid activation and L2 regularization.
    Useful for understanding the underlying mathematics and educational purposes.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 regularization: str = 'l2', lambda_: float = 0.01):
        """
        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of training iterations
            regularization: Type of regularization ('l2' or 'none')
            lambda_: Regularization strength
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.theta = None
        self.loss_history = []
        
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function: 1 / (1 + exp(-z))"""
        return sigmoid(z)
    
    def compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """Compute binary cross-entropy loss with optional regularization."""
        m = X.shape[0]
        z = X @ theta
        h = self.sigmoid(z)
        
        # Clip to avoid log(0)
        h = np.clip(h, 1e-15, 1 - 1e-15)
        
        # Binary cross-entropy loss
        cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        
        # Add regularization (excluding bias term)
        if self.regularization == 'l2':
            reg_cost = (self.lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
            cost += reg_cost
        
        return cost
    
    def compute_gradient(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute gradient of loss with respect to theta."""
        m = X.shape[0]
        z = X @ theta
        h = self.sigmoid(z)
        
        gradient = (X.T @ (h - y)) / m
        
        # Add regularization gradient (excluding bias term)
        if self.regularization == 'l2':
            gradient[1:] += (self.lambda_ / m) * theta[1:]
        
        return gradient
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train logistic regression model using gradient descent."""
        m, n = X.shape
        self.theta = np.zeros(n)
        self.loss_history = []
        
        for iteration in range(self.n_iterations):
            # Compute gradient
            gradient = self.compute_gradient(X, y, self.theta)
            
            # Update parameters
            self.theta -= self.learning_rate * gradient
            
            # Record loss
            cost = self.compute_cost(X, y, self.theta)
            self.loss_history.append(cost)
            
            if (iteration + 1) % max(1, self.n_iterations // 10) == 0:
                print(f"  Iteration {iteration + 1}/{self.n_iterations}, Loss: {cost:.6f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class."""
        return self.sigmoid(X @ self.theta)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def get_params(self) -> np.ndarray:
        """Return learned parameters."""
        return self.theta.copy()


class LogisticRegressionPipeline:
    """End-to-end logistic regression pipeline for sentiment analysis."""
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, "models")
        self.viz_dir = os.path.join(output_dir, "visualizations")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.logger = self._setup_logger()
        
        # Pipeline components
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.sklearn_model = None
        self.custom_model = None
        self.scaler = None
        
        self.feature_names = ['bias', 'positive', 'negative']
        self.preprocessing_config = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger("logistic_regression")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            fh = logging.FileHandler(
                os.path.join(self.model_dir, "training.log")
            )
            ch = logging.StreamHandler()
            
            fmt = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(fmt)
            ch.setFormatter(fmt)
            
            logger.addHandler(fh)
            logger.addHandler(ch)
        
        return logger
    
    def extract_features(self, tweets: List[str], labels: np.ndarray,
                        freqs: Dict[Tuple[str, int], int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract logistic regression features from tweets.
        
        Features:
        - Bias term (always 1)
        - Sum of positive word frequencies in tweet
        - Sum of negative word frequencies in tweet
        
        Args:
            tweets: List of raw tweet strings
            labels: Binary labels (0 for negative, 1 for positive)
            freqs: Frequency dictionary from build_freqs
        
        Returns:
            X: Feature matrix of shape (n_samples, 3)
            y: Label vector
        """
        self.logger.info(f"Extracting features from {len(tweets)} tweets...")
        
        X = np.zeros((len(tweets), 3))
        preprocessor = TweetPreprocessor()
        
        for i, tweet in enumerate(tqdm(tweets, desc="Extracting features")):
            # Extract tokens
            tokens = preprocessor.process(tweet)
            
            # Sum up positive and negative word frequencies
            for token in tokens:
                pos_count = freqs.get((token, 1), 0)
                neg_count = freqs.get((token, 0), 0)
                X[i, 1] += pos_count  # Positive sum
                X[i, 2] += neg_count  # Negative sum
            
            X[i, 0] = 1  # Bias term
        
        # Optional: Normalize features (excluding bias)
        if USE_NORMALIZED_FEATURES:
            pos_mean = np.mean(X[:, 1])
            pos_std = np.std(X[:, 1])
            neg_mean = np.mean(X[:, 2])
            neg_std = np.std(X[:, 2])
            
            X[:, 1] = (X[:, 1] - pos_mean) / (pos_std + 1e-8)
            X[:, 2] = (X[:, 2] - neg_mean) / (neg_std + 1e-8)
            
            self.preprocessing_config = {
                'pos_mean': float(pos_mean),
                'pos_std': float(pos_std),
                'neg_mean': float(neg_mean),
                'neg_std': float(neg_std)
            }
        
        self.logger.info(f"Features shape: {X.shape}")
        self.logger.info(f"Feature statistics:\n{self._get_feature_stats(X)}")
        
        return X, labels
    
    def _get_feature_stats(self, X: np.ndarray) -> str:
        """Generate feature statistics summary."""
        stats = []
        for i, name in enumerate(self.feature_names):
            stats.append(f"  {name:10s}: min={X[:, i].min():.4f}, "
                        f"max={X[:, i].max():.4f}, "
                        f"mean={X[:, i].mean():.4f}, "
                        f"std={X[:, i].std():.4f}")
        return "\n".join(stats)
    
    def train_test_split_data(self, X: np.ndarray, y: np.ndarray,
                             test_size: float = 1 - TRAIN_TEST_SPLIT) -> None:
        """Split data into training and test sets with stratification."""
        self.logger.info(f"Splitting data: {TRAIN_TEST_SPLIT:.1%} train, {test_size:.1%} test")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        self.logger.info(f"Train set: {self.X_train.shape[0]} samples "
                        f"(positive: {np.sum(self.y_train)}, "
                        f"negative: {len(self.y_train) - np.sum(self.y_train)})")
        self.logger.info(f"Test set: {self.X_test.shape[0]} samples "
                        f"(positive: {np.sum(self.y_test)}, "
                        f"negative: {len(self.y_test) - np.sum(self.y_test)})")
    
    def train_sklearn_model(self) -> Dict[str, float]:
        """Train scikit-learn logistic regression with cross-validation."""
        self.logger.info("Training scikit-learn LogisticRegression...")
        
        self.sklearn_model = SklearnLogisticRegression(
            solver=SOLVER,
            max_iter=MAX_ITER,
            penalty=PENALTY,
            C=C,
            random_state=RANDOM_SEED,
            verbose=0
        )
        
        # Train on training set
        self.sklearn_model.fit(self.X_train, self.y_train)
        
        # Cross-validation on training set
        cv_scores = cross_val_score(
            self.sklearn_model,
            self.X_train,
            self.y_train,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
            scoring='roc_auc'
        )
        
        self.logger.info(f"Cross-validation scores: {cv_scores}")
        self.logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Evaluate on test set
        metrics = self._evaluate_model(self.sklearn_model, "Scikit-Learn Model")
        
        return metrics
    
    def train_custom_model(self) -> Dict[str, float]:
        """Train custom logistic regression implementation."""
        self.logger.info("Training custom LogisticRegression with gradient descent...")
        
        self.custom_model = CustomLogisticRegression(
            learning_rate=0.001,
            n_iterations=5000,
            regularization='l2',
            lambda_=0.01
        )
        
        self.custom_model.fit(self.X_train, self.y_train)
        
        metrics = self._evaluate_model(self.custom_model, "Custom Model")
        
        return metrics
    
    def _evaluate_model(self, model, model_name: str) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Evaluating {model_name}")
        self.logger.info(f"{'='*60}")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # For sklearn models, predict_proba returns (n_samples, 2) array
        # Extract probability of positive class
        if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        self.logger.info(f"Accuracy:  {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall:    {recall:.4f}")
        self.logger.info(f"F1-Score:  {f1:.4f}")
        self.logger.info(f"ROC-AUC:   {roc_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        self.logger.info(f"\nConfusion Matrix:\n{cm}")
        
        # Classification report
        self.logger.info(f"\nClassification Report:\n{classification_report(self.y_test, y_pred)}")
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist()
        }
    
    def visualize_decision_boundary(self) -> None:
        """Create professional scatter plot with decision boundaries."""
        self.logger.info("Generating decision boundary visualization...")
        
        models = {
            'sklearn': self.sklearn_model,
            'custom': self.custom_model
        }
        
        for model_name, model in models.items():
            if model is None:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Get model parameters
            if hasattr(model, 'coef_'):
                theta = np.concatenate([[model.intercept_[0]], model.coef_[0]])
            else:
                theta = model.get_params()
            
            # Scatter plot: positive tweets in green, negative in red
            colors = ['red', 'green']
            for label, color in enumerate(colors):
                mask = self.y_test == label
                ax.scatter(
                    self.X_test[mask, 1],
                    self.X_test[mask, 2],
                    c=color,
                    alpha=0.6,
                    s=30,
                    label=f"{'Positive' if label else 'Negative'} tweets",
                    edgecolors='black',
                    linewidth=0.3
                )
            
            # Decision boundary: theta[0] + theta[1]*pos + theta[2]*neg = 0
            # neg = (-theta[0] - theta[1]*pos) / theta[2]
            pos_range = np.linspace(
                self.X_test[:, 1].min() - 1,
                self.X_test[:, 1].max() + 1,
                300
            )
            
            # Avoid division by zero
            if abs(theta[2]) > 1e-10:
                neg_boundary = (-theta[0] - theta[1] * pos_range) / theta[2]
                ax.plot(pos_range, neg_boundary, 'b-', linewidth=2.5, label='Decision Boundary')
            
            # Directional arrows showing classification direction
            offset_pos = np.percentile(self.X_test[:, 1], 25)
            offset_neg = (-theta[0] - theta[1] * offset_pos) / theta[2]
            
            # Direction of positive class (perpendicular to boundary)
            if abs(theta[1]) > 1e-10:
                direction = offset_pos * theta[2] / theta[1]
                
                # Green arrow (positive direction)
                ax.arrow(
                    offset_pos, offset_neg,
                    offset_pos * 0.3, direction * 0.3,
                    head_width=0.15, head_length=0.1,
                    fc='green', ec='darkgreen', alpha=0.7, linewidth=2
                )
                
                # Red arrow (negative direction)
                ax.arrow(
                    offset_pos, offset_neg,
                    -offset_pos * 0.3, -direction * 0.3,
                    head_width=0.15, head_length=0.1,
                    fc='red', ec='darkred', alpha=0.7, linewidth=2
                )
            
            # Prediction confidence contours
            pos_mesh = np.linspace(self.X_test[:, 1].min(), self.X_test[:, 1].max(), 100)
            neg_mesh = np.linspace(self.X_test[:, 2].min(), self.X_test[:, 2].max(), 100)
            X_mesh, Y_mesh = np.meshgrid(pos_mesh, neg_mesh)
            
            Z = np.zeros_like(X_mesh)
            for i in range(X_mesh.shape[0]):
                for j in range(X_mesh.shape[1]):
                    X_point = np.array([[1, X_mesh[i, j], Y_mesh[i, j]]])
                    proba = model.predict_proba(X_point)
                    # Handle sklearn's 2D output vs custom model's 1D
                    if isinstance(proba, np.ndarray) and proba.ndim == 2:
                        Z[i, j] = proba[0, 1]  # Probability of positive class
                    else:
                        Z[i, j] = proba[0]
            
            contour = ax.contour(X_mesh, Y_mesh, Z, levels=[0.5], colors='blue',
                                linewidths=2, linestyles='--', alpha=0.5)
            ax.clabel(contour, inline=True, fontsize=8)
            
            # Formatting
            ax.set_xlabel('Positive Word Sum', fontsize=12, fontweight='bold')
            ax.set_ylabel('Negative Word Sum', fontsize=12, fontweight='bold')
            ax.set_title(f'Logistic Regression Decision Boundary\n({model_name.upper()} Model)',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Save and display
            output_path = os.path.join(self.viz_dir, f'decision_boundary_{model_name}.png')
            fig.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved to {output_path}")
            plt.close(fig)
    
    def visualize_roc_curves(self) -> None:
        """Plot ROC curves for both models."""
        self.logger.info("Generating ROC curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        models = {
            'sklearn': self.sklearn_model,
            'custom': self.custom_model
        }
        
        for model_name, model in models.items():
            if model is None:
                continue
            
            y_pred_proba = model.predict_proba(self.X_test)
            
            # Handle sklearn's 2D output
            if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
                y_pred_proba = y_pred_proba[:, 1]
            
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            ax.plot(fpr, tpr, linewidth=2.5,
                   label=f'{model_name.upper()} (AUC={roc_auc:.4f})')
        
        # Random classifier baseline
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.viz_dir, 'roc_curves.png')
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved to {output_path}")
        plt.close(fig)
    
    def visualize_training_loss(self) -> None:
        """Plot custom model training loss."""
        if self.custom_model is None or not self.custom_model.loss_history:
            return
        
        self.logger.info("Generating training loss visualization...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.custom_model.loss_history, linewidth=2, color='darkblue')
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss (Binary Cross-Entropy)', fontsize=12, fontweight='bold')
        ax.set_title('Custom Model Training Loss Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.viz_dir, 'training_loss.png')
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved to {output_path}")
        plt.close(fig)
    
    def save_models(self) -> None:
        """Persist trained models to disk."""
        self.logger.info("Saving trained models...")
        
        if self.sklearn_model:
            sklearn_path = os.path.join(self.model_dir, 'sklearn_logistic_model.pkl')
            with open(sklearn_path, 'wb') as f:
                pickle.dump(self.sklearn_model, f)
            self.logger.info(f"Saved sklearn model to {sklearn_path}")
        
        if self.custom_model:
            custom_path = os.path.join(self.model_dir, 'custom_logistic_model.pkl')
            with open(custom_path, 'wb') as f:
                pickle.dump(self.custom_model, f)
            self.logger.info(f"Saved custom model to {custom_path}")
    
    def save_metadata(self, metrics_sklearn: Dict, metrics_custom: Dict) -> None:
        """Save model metadata and evaluation results."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'solver': SOLVER,
                'max_iterations': MAX_ITER,
                'penalty': PENALTY,
                'regularization_C': C,
                'train_test_split': TRAIN_TEST_SPLIT,
                'cv_folds': CV_FOLDS,
                'use_normalized_features': USE_NORMALIZED_FEATURES,
            },
            'preprocessing': self.preprocessing_config,
            'train_set_size': int(self.X_train.shape[0]),
            'test_set_size': int(self.X_test.shape[0]),
            'feature_names': self.feature_names,
            'sklearn_metrics': metrics_sklearn,
            'custom_metrics': metrics_custom,
        }
        
        metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved metadata to {metadata_path}")
    
    def run_pipeline(self) -> None:
        """Execute complete logistic regression pipeline."""
        self.logger.info("="*70)
        self.logger.info("STARTING LOGISTIC REGRESSION PIPELINE")
        self.logger.info("="*70)
        
        try:
            # Load NLTK data
            self.logger.info("Loading NLTK Twitter samples...")
            all_positive_tweets = twitter_samples.strings("positive_tweets.json")
            all_negative_tweets = twitter_samples.strings("negative_tweets.json")
            
            # Use first 4000 of each for training (as per lab)
            train_pos = all_positive_tweets[:4000]
            train_neg = all_negative_tweets[:4000]
            tweets = train_pos + train_neg
            labels = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
            
            self.logger.info(f"Loaded {len(tweets)} tweets")
            
            # Build frequency dictionary
            self.logger.info("Building frequency dictionary...")
            freqs = build_freqs(tweets, labels)
            self.logger.info(f"Frequency dictionary size: {len(freqs)}")
            
            # Extract features
            X, y = self.extract_features(tweets, labels, freqs)
            
            # Train-test split
            self.train_test_split_data(X, y)
            
            # Train models
            metrics_sklearn = self.train_sklearn_model()
            metrics_custom = self.train_custom_model()
            
            # Visualizations
            self.visualize_decision_boundary()
            self.visualize_roc_curves()
            self.visualize_training_loss()
            
            # Save outputs
            self.save_models()
            self.save_metadata(metrics_sklearn, metrics_custom)
            
            self.logger.info("="*70)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*70)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point."""
    try:
        pipeline = LogisticRegressionPipeline(output_dir=OUTPUT_DIR)
        pipeline.run_pipeline()
        print("\n✓ Logistic regression pipeline completed successfully!")
        print(f"✓ Models saved to: {pipeline.model_dir}")
        print(f"✓ Visualizations saved to: {pipeline.viz_dir}")
    except Exception as e:
        print(f"✗ Pipeline failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
