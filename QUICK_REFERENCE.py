#!/usr/bin/env python3
"""
QUICK REFERENCE GUIDE - Logistic Regression Sentiment Analysis Pipeline
========================================================================

This guide provides quick snippets for common tasks with the trained models.
"""

# ============================================================================
# 1. SINGLE TWEET PREDICTION
# ============================================================================

from model_inference import LogisticRegressionInference

inference = LogisticRegressionInference()

# Predict using sklearn model
result = inference.predict("I love this product!", model='sklearn')
print(f"Sentiment: {result['sentiment']}")          # Output: positive
print(f"Confidence: {result['confidence']:.4f}")    # Output: 0.9999

# ============================================================================
# 2. BATCH PREDICTIONS
# ============================================================================

tweets = [
    "Amazing experience!",
    "Terrible service",
    "Not bad"
]

results = inference.batch_predict(tweets, model='sklearn')
for r in results:
    print(f"{r['tweet']:30s} → {r['sentiment']:10s} ({r['confidence']:.3f})")

# ============================================================================
# 3. EXTRACT FEATURES FROM TWEET
# ============================================================================

# Get feature vector for a tweet
tweet = "I really enjoyed this!"
features = inference.extract_features(tweet)
print(f"Bias term: {features[0, 0]}")           # 1.0
print(f"Positive words sum: {features[0, 1]}")  # Normalized
print(f"Negative words sum: {features[0, 2]}")  # Normalized

# ============================================================================
# 4. MODEL COEFFICIENTS & INTERPRETATION
# ============================================================================

coefs = inference.get_model_coefficients()
print(f"Intercept: {coefs['intercept']:.6f}")
print(f"Positive words coefficient: {coefs['positive_words_coef']:.6f}")
print(f"Negative words coefficient: {coefs['negative_words_coef']:.6f}")

# Decision boundary: neg = (-intercept - pos*coef_pos) / coef_neg

# ============================================================================
# 5. COMPARE BOTH MODELS
# ============================================================================

tweet = "I absolutely love this! Best product ever!"

sklearn_pred = inference.predict(tweet, 'sklearn')
custom_pred = inference.predict(tweet, 'custom')

print(f"Tweet: {tweet}")
print(f"SKlearn Result:  {sklearn_pred['sentiment']:10s} ({sklearn_pred['confidence']:.4f})")
print(f"Custom Result:   {custom_pred['sentiment']:10s}  ({custom_pred['confidence']:.4f})")

# ============================================================================
# 6. ANALYZE DECISION CONFIDENCE
# ============================================================================

test_cases = [
    ("Love it!", "high_positive"),
    ("Not sure", "neutral"),
    ("Hate it!", "high_negative"),
]

for tweet, expected in test_cases:
    result = inference.predict(tweet, 'sklearn')
    confidence = result['confidence']
    prediction = "POSITIVE" if confidence > 0.5 else "NEGATIVE"
    
    print(f"{tweet:20s} → {prediction:10s} (confidence: {confidence:.4f})")

# ============================================================================
# 7. LOAD METADATA & MODEL INFO
# ============================================================================

import json

with open('preprocessed_data/models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Training set size: {metadata['train_set_size']}")
print(f"Test set size: {metadata['test_set_size']}")
print(f"Sklearn accuracy: {metadata['sklearn_metrics']['accuracy']:.4f}")
print(f"Custom accuracy: {metadata['custom_metrics']['accuracy']:.4f}")

# ============================================================================
# 8. FEATURE STATISTICS
# ============================================================================

preprocessing = metadata['preprocessing']
print(f"Positive words - Mean: {preprocessing['pos_mean']:.2f}, Std: {preprocessing['pos_std']:.2f}")
print(f"Negative words - Mean: {preprocessing['neg_mean']:.2f}, Std: {preprocessing['neg_std']:.2f}")

# ============================================================================
# 9. DIRECTLY LOAD MODELS (Advanced)
# ============================================================================

import pickle
import numpy as np

# Load sklearn model
with open('preprocessed_data/models/sklearn_logistic_model.pkl', 'rb') as f:
    sklearn_model = pickle.load(f)

# Load custom model
with open('preprocessed_data/models/custom_logistic_model.pkl', 'rb') as f:
    custom_model = pickle.load(f)

# Make prediction with raw feature vector [bias, pos_sum, neg_sum]
X_test = np.array([[1, 2.5, -0.8]])  # Example feature vector
y_pred = sklearn_model.predict(X_test)
y_proba = sklearn_model.predict_proba(X_test)

print(f"Prediction: {y_pred[0]}")              # 1 = positive, 0 = negative
print(f"Probabilities: {y_proba[0]}")          # [prob_negative, prob_positive]

# ============================================================================
# 10. PRINT DETAILED MODEL INFO
# ============================================================================

inference.print_model_info()

# ============================================================================
# KEY MODEL INFORMATION
# ============================================================================

"""
SKLEARN MODEL:
  - Solver: LBFGS
  - Accuracy: 99.69%
  - Precision: 99.38%
  - Recall: 100%
  - ROC-AUC: 100%
  
CUSTOM MODEL:
  - Gradient Descent with 5000 iterations
  - Accuracy: 96.13%
  - Precision: 92.81%
  - Recall: 100%
  - ROC-AUC: 100%

FEATURES:
  - Bias term (always 1)
  - Sum of positive word frequencies (normalized)
  - Sum of negative word frequencies (normalized)

OUTPUTS:
  - Models: preprocessed_data/models/
  - Visualizations: preprocessed_data/visualizations/
  - Logs: preprocessed_data/models/training.log
"""

# ============================================================================
# COMMON PATTERNS
# ============================================================================

# Pattern 1: Get sentiment with confidence threshold
def predict_with_confidence(tweet, threshold=0.8):
    result = inference.predict(tweet, 'sklearn')
    if result['confidence'] < threshold:
        return "uncertain"
    return result['sentiment']

# Pattern 2: Batch process and filter by confidence
def batch_predict_confident(tweets, confidence_threshold=0.9):
    results = inference.batch_predict(tweets, 'sklearn')
    return [r for r in results if r['confidence'] >= confidence_threshold]

# Pattern 3: Compare model agreement
def get_model_agreement(tweet):
    sklearn_result = inference.predict(tweet, 'sklearn')
    custom_result = inference.predict(tweet, 'custom')
    agree = sklearn_result['sentiment'] == custom_result['sentiment']
    return {
        'agree': agree,
        'sklearn': sklearn_result['sentiment'],
        'custom': custom_result['sentiment'],
        'sklearn_conf': sklearn_result['confidence'],
        'custom_conf': custom_result['confidence']
    }

# ============================================================================
# DECISION BOUNDARY EQUATION
# ============================================================================

"""
Linear Decision Boundary:
  z = θ₀ + θ₁*positive_sum + θ₂*negative_sum = 0

  If z > 0  → Predict POSITIVE
  If z < 0  → Predict NEGATIVE

To find the boundary line (plotting negative_sum vs positive_sum):
  negative_sum = (-θ₀ - θ₁*positive_sum) / θ₂

Example with actual coefficients:
  θ₀ = 0.000000
  θ₁ = 0.000538
  θ₂ = -0.000558

  neg = (0 - 0.000538*pos) / (-0.000558)
  neg = 0.964*pos
"""

# ============================================================================
# FILES REFERENCE
# ============================================================================

"""
Core Files:
  - logistic_regression_model.py      → Main pipeline
  - model_inference.py                → Inference interface
  - run_complete_pipeline.py          → Full orchestration

Documentation:
  - LOGISTIC_REGRESSION_GUIDE.md      → Detailed guide
  - IMPLEMENTATION_SUMMARY.md         → Complete summary
  - QUICK_REFERENCE.md                → This file

Generated Models:
  - preprocessed_data/models/sklearn_logistic_model.pkl
  - preprocessed_data/models/custom_logistic_model.pkl
  - preprocessed_data/models/model_metadata.json

Visualizations:
  - preprocessed_data/visualizations/decision_boundary_sklearn.png
  - preprocessed_data/visualizations/decision_boundary_custom.png
  - preprocessed_data/visualizations/roc_curves.png
  - preprocessed_data/visualizations/training_loss.png
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Q: Model predicts everything as positive
A: Lower the decision threshold from 0.5 to 0.3
   prediction = "positive" if probability > 0.3 else "negative"

Q: Too many false positives
A: Use the custom model (higher precision)
   or increase decision threshold to 0.7

Q: Need more confident predictions
A: Filter by confidence > 0.95
   result = predict(tweet)
   if result['confidence'] < 0.95: return "uncertain"

Q: Want to understand which words matter
A: Extract features and check sums
   features = extract_features(tweet)
   if features[0, 1] > features[0, 2]: "More positive words"
   
Q: Model performance degradation
A: Check if new data distribution matches training data
   Compare feature statistics
   Monitor prediction confidence distribution
"""

# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

"""
Test Set Performance (n=1,600):
  
  Sklearn Model:
    - Accuracy:    99.69%
    - Precision:   99.38% (99% of predicted positive are correct)
    - Recall:      100.00% (finds all positive tweets)
    - False Positives: 5/800 (0.625%)
    - False Negatives: 0/800 (0%)
  
  Custom Model:
    - Accuracy:    96.13%
    - Precision:   92.81% (higher false positives)
    - Recall:      100.00% (still finds all positive)
    - False Positives: 62/800 (7.75%)
    - False Negatives: 0/800 (0%)

Recommendation:
  - Use sklearn for high precision (minimize false positives)
  - Use custom for high recall (catch all positives)
"""
