#!/usr/bin/env python3
"""
Model Prediction and Inference Guide
=====================================

This script demonstrates how to:
1. Load trained logistic regression models
2. Make predictions on new tweets
3. Analyze decision confidence and probabilities
4. Understand feature importance and model interpretation

Key Features:
- Load both sklearn and custom logistic regression models
- Preprocess new tweets using the same pipeline
- Generate predictions with confidence scores
- Visualize decision confidence across feature space
- Extract and interpret model coefficients
"""

import pickle
import json
import numpy as np
from pathlib import Path

from tweet_preprocessing import TweetPreprocessor, build_freqs
from nltk.corpus import twitter_samples


class LogisticRegressionInference:
    """Load and use trained logistic regression models for inference."""
    
    def __init__(self, model_dir: str = "preprocessed_data/models",
                 metadata_file: str = "preprocessed_data/models/model_metadata.json"):
        """
        Initialize inference pipeline.
        
        Args:
            model_dir: Directory containing trained models
            metadata_file: Path to model metadata JSON
        """
        self.model_dir = model_dir
        self.preprocessor = TweetPreprocessor()
        
        # Load models
        self.sklearn_model = self._load_model(f"{model_dir}/sklearn_logistic_model.pkl")
        self.custom_model = self._load_model(f"{model_dir}/custom_logistic_model.pkl")
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.preprocessing_config = self.metadata.get('preprocessing', {})
        print("✓ Models and metadata loaded successfully")
    
    def _load_model(self, path: str):
        """Load pickled model."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Model not found at {path}")
            return None
    
    def extract_features(self, tweet: str) -> np.ndarray:
        """
        Extract sentiment features from a single tweet.
        
        Features: [bias, positive_sum, negative_sum]
        
        Args:
            tweet: Raw tweet text
        
        Returns:
            Feature vector of shape (1, 3)
        """
        # Load frequency dictionary
        all_positive_tweets = twitter_samples.strings("positive_tweets.json")
        all_negative_tweets = twitter_samples.strings("negative_tweets.json")
        train_pos = all_positive_tweets[:4000]
        train_neg = all_negative_tweets[:4000]
        tweets = train_pos + train_neg
        labels = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
        freqs = build_freqs(tweets, labels)
        
        # Extract tokens
        tokens = self.preprocessor.process(tweet)
        
        # Initialize feature vector
        X = np.zeros((1, 3))
        X[0, 0] = 1  # Bias term
        
        # Sum positive and negative word frequencies
        for token in tokens:
            X[0, 1] += freqs.get((token, 1), 0)  # Positive sum
            X[0, 2] += freqs.get((token, 0), 0)  # Negative sum
        
        # Normalize features if configured
        if self.preprocessing_config:
            pos_mean = self.preprocessing_config.get('pos_mean', 0)
            pos_std = self.preprocessing_config.get('pos_std', 1)
            neg_mean = self.preprocessing_config.get('neg_mean', 0)
            neg_std = self.preprocessing_config.get('neg_std', 1)
            
            X[0, 1] = (X[0, 1] - pos_mean) / (pos_std + 1e-8)
            X[0, 2] = (X[0, 2] - neg_mean) / (neg_std + 1e-8)
        
        return X
    
    def predict(self, tweet: str, model: str = 'sklearn') -> dict:
        """
        Predict sentiment for a tweet.
        
        Args:
            tweet: Tweet text to analyze
            model: Which model to use ('sklearn' or 'custom')
        
        Returns:
            Dictionary with prediction, confidence, and feature info
        """
        if model == 'sklearn':
            if self.sklearn_model is None:
                return {"error": "sklearn model not loaded"}
            classifier = self.sklearn_model
        else:
            if self.custom_model is None:
                return {"error": "custom model not loaded"}
            classifier = self.custom_model
        
        # Extract features
        X = self.extract_features(tweet)
        
        # Make prediction
        prediction = classifier.predict(X)[0]
        proba = classifier.predict_proba(X)
        
        # Handle sklearn's 2D output
        if isinstance(proba, np.ndarray) and proba.ndim == 2:
            confidence = proba[0, 1]
        else:
            confidence = proba[0]
        
        result = {
            'tweet': tweet,
            'model': model,
            'sentiment': 'positive' if prediction == 1 else 'negative',
            'prediction': int(prediction),
            'confidence': float(confidence),
            'positive_words_sum': float(X[0, 1]),
            'negative_words_sum': float(X[0, 2]),
            'raw_score': float(proba[0, 1] if isinstance(proba, np.ndarray) and proba.ndim == 2 else proba[0])
        }
        
        return result
    
    def batch_predict(self, tweets: list, model: str = 'sklearn') -> list:
        """
        Predict sentiment for multiple tweets.
        
        Args:
            tweets: List of tweet texts
            model: Which model to use
        
        Returns:
            List of prediction results
        """
        results = []
        for tweet in tweets:
            results.append(self.predict(tweet, model))
        return results
    
    def get_model_coefficients(self) -> dict:
        """
        Extract and display model coefficients.
        
        Returns:
            Dictionary with coefficient values and interpretations
        """
        if self.sklearn_model is None:
            return {"error": "sklearn model not loaded"}
        
        intercept = self.sklearn_model.intercept_[0]
        coefs = self.sklearn_model.coef_[0]
        
        return {
            'intercept': float(intercept),
            'bias_coef': float(coefs[0]),
            'positive_words_coef': float(coefs[1]),
            'negative_words_coef': float(coefs[2]),
            'interpretation': {
                'positive_words': 'Increases positive sentiment' if coefs[1] > 0 else 'Decreases positive sentiment',
                'negative_words': 'Decreases positive sentiment' if coefs[2] < 0 else 'Increases positive sentiment'
            }
        }
    
    def analyze_examples(self):
        """Analyze some example tweets."""
        examples = [
            "This is amazing! I love it! 😍",
            "Terrible product, very disappointed 😞",
            "Just okay, nothing special",
            "Best day ever! Feeling great! #blessed",
            "Worst experience of my life 😤",
        ]
        
        print("\n" + "="*80)
        print("EXAMPLE PREDICTIONS")
        print("="*80)
        
        for tweet in examples:
            sklearn_result = self.predict(tweet, 'sklearn')
            custom_result = self.predict(tweet, 'custom')
            
            print(f"\nTweet: {tweet}")
            print(f"  SKlearn:     {sklearn_result['sentiment']:8s} (confidence: {sklearn_result['confidence']:.4f})")
            print(f"  Custom:      {custom_result['sentiment']:8s} (confidence: {custom_result['confidence']:.4f})")
            print(f"  Features:    positive={sklearn_result['positive_words_sum']:.2f}, "
                  f"negative={sklearn_result['negative_words_sum']:.2f}")
    
    def print_model_info(self):
        """Print information about trained models."""
        print("\n" + "="*80)
        print("MODEL INFORMATION")
        print("="*80)
        
        config = self.metadata.get('configuration', {})
        print(f"\nTraining Configuration:")
        print(f"  Solver: {config.get('solver', 'N/A')}")
        print(f"  Max iterations: {config.get('max_iterations', 'N/A')}")
        print(f"  Penalty type: {config.get('penalty', 'N/A')}")
        print(f"  Regularization C: {config.get('regularization_C', 'N/A')}")
        
        train_size = self.metadata.get('train_set_size', 0)
        test_size = self.metadata.get('test_set_size', 0)
        print(f"\nData Split:")
        print(f"  Training set: {train_size} samples")
        print(f"  Test set: {test_size} samples")
        
        sklearn_metrics = self.metadata.get('sklearn_metrics', {})
        custom_metrics = self.metadata.get('custom_metrics', {})
        
        print(f"\nScikit-learn Model Performance:")
        print(f"  Accuracy:  {sklearn_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  Precision: {sklearn_metrics.get('precision', 'N/A'):.4f}")
        print(f"  Recall:    {sklearn_metrics.get('recall', 'N/A'):.4f}")
        print(f"  F1-Score:  {sklearn_metrics.get('f1', 'N/A'):.4f}")
        print(f"  ROC-AUC:   {sklearn_metrics.get('roc_auc', 'N/A'):.4f}")
        
        print(f"\nCustom Model Performance:")
        print(f"  Accuracy:  {custom_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  Precision: {custom_metrics.get('precision', 'N/A'):.4f}")
        print(f"  Recall:    {custom_metrics.get('recall', 'N/A'):.4f}")
        print(f"  F1-Score:  {custom_metrics.get('f1', 'N/A'):.4f}")
        print(f"  ROC-AUC:   {custom_metrics.get('roc_auc', 'N/A'):.4f}")
        
        # Model coefficients
        coefs = self.get_model_coefficients()
        print(f"\nModel Coefficients (SKlearn):")
        print(f"  Intercept: {coefs['intercept']:.6f}")
        print(f"  Bias term: {coefs['bias_coef']:.6f}")
        print(f"  Positive words: {coefs['positive_words_coef']:.6f} - {coefs['interpretation']['positive_words']}")
        print(f"  Negative words: {coefs['negative_words_coef']:.6f} - {coefs['interpretation']['negative_words']}")


def main():
    """Main inference example."""
    import nltk
    from nltk.corpus import twitter_samples
    
    # Ensure NLTK data is available
    try:
        nltk.data.find('corpora/twitter_samples')
    except LookupError:
        nltk.download('twitter_samples')
    
    # Initialize inference pipeline
    inference = LogisticRegressionInference()
    
    # Print model information
    inference.print_model_info()
    
    # Analyze example tweets
    inference.analyze_examples()
    
    # Interactive prediction
    print("\n" + "="*80)
    print("INTERACTIVE SENTIMENT PREDICTION")
    print("="*80)
    print("Enter tweets to analyze (type 'quit' to exit, 'examples' for more examples)")
    
    while True:
        user_input = input("\nEnter a tweet: ").strip()
        
        if user_input.lower() == 'quit':
            print("Exiting...")
            break
        elif user_input.lower() == 'examples':
            inference.analyze_examples()
        elif user_input:
            sklearn_result = inference.predict(user_input, 'sklearn')
            custom_result = inference.predict(user_input, 'custom')
            
            print(f"\n✓ Analysis Results:")
            print(f"  SKlearn Model:  {sklearn_result['sentiment'].upper()} "
                  f"(confidence: {sklearn_result['confidence']:.4f})")
            print(f"  Custom Model:   {custom_result['sentiment'].upper()} "
                  f"(confidence: {custom_result['confidence']:.4f})")
            print(f"  Features Extracted:")
            print(f"    - Positive words sum: {sklearn_result['positive_words_sum']:.2f}")
            print(f"    - Negative words sum: {sklearn_result['negative_words_sum']:.2f}")


if __name__ == '__main__':
    main()
