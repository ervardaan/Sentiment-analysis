""" Logistic Regression Sentiment Analysis Pipeline - Complete Documentation
==============================================================================

## Overview

This document provides comprehensive documentation for the robust logistic regression
sentiment analysis pipeline. The pipeline includes state-of-the-art implementations,
advanced visualizations, and production-ready model serialization.

## Pipeline Architecture

### 1. Preprocessing Module (`tweet_preprocessing.py`)
- **TweetPreprocessor**: Handles tweet tokenization, stopword removal, and stemming
- **TweetVectorizer**: TF-IDF vectorization with configurable parameters
- Processes 10,000 tweets with ~13K baseline tokens -> ~6.7K processed tokens
- Outputs: Vectors, tokens, metadata, and feature names

### 2. Logistic Regression Module (`logistic_regression_model.py`)

#### CustomLogisticRegression
A from-scratch implementation using gradient descent and sigmoid activation:
- **Learning mechanism**: Batch gradient descent with configurable learning rate
- **Activation**: Sigmoid function: σ(z) = 1 / (1 + e^(-z))
- **Loss function**: Binary cross-entropy with L2 regularization
- **Optimization**: Adaptive learning with gradient-based updates
- **Cost function**: 
  ```
  J(θ) = -1/m Σ[y*log(h) + (1-y)*log(1-h)] + λ/(2m) ||θ[1:]||²
  ```

#### SklearnLogisticRegression
Production-grade implementation from scikit-learn:
- **Solver**: LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
- **Regularization**: L2 penalty with C=1.0
- **Convergence**: max_iter=1000
- **Cross-validation**: 5-fold stratified CV on training set

### 3. Feature Extraction

For each tweet, extract 3 features:
1. **Bias term**: Always = 1
2. **Positive words sum**: Sum of frequencies of positive-sentiment words in tweet
3. **Negative words sum**: Sum of frequencies of negative-sentiment words in tweet

Example feature vector:
```
[bias=1.0, pos_sum=5.2, neg_sum=-0.8]
```

Features are z-score normalized:
```
x_norm = (x - mean(x)) / std(x)
```

### 4. Decision Boundary

The logistic regression model creates a linear decision boundary:
```
z = θ₀ + θ₁*pos + θ₂*neg = 0

Where:
- z > 0  => Predict positive sentiment (y=1)
- z < 0  => Predict negative sentiment (y=0)

Solved for negative axis:
neg = (-θ₀ - θ₁*pos) / θ₂
```

The perpendicular direction to the boundary is:
```
direction = pos * θ₂ / θ₁
```

## Model Performance

### Scikit-learn Model
```
Accuracy:  99.69%
Precision: 99.38%
Recall:    100.00%
F1-Score:  99.69%
ROC-AUC:   100.00%

Confusion Matrix:
[[795   5]
 [  0 800]]
```

- **True Negatives**: 795/800 (negative tweets correctly classified)
- **False Positives**: 5/800 (negative tweets misclassified as positive)
- **False Negatives**: 0/800 (positive tweets misclassified as negative)
- **True Positives**: 800/800 (positive tweets correctly classified)

### Custom Model
```
Accuracy:  96.13%
Precision: 92.81%
Recall:    100.00%
F1-Score:  96.27%
ROC-AUC:   100.00%

Confusion Matrix:
[[738  62]
 [  0 800]]
```

- More conservative than sklearn (more false positives)
- Perfect recall on positive class
- Slightly lower precision due to gradient descent convergence

### Cross-Validation Results
```
Fold 1: 0.9994
Fold 2: 1.0000
Fold 3: 0.9994
Fold 4: 0.9993
Fold 5: 0.9995

Mean: 0.9995 ± 0.0002
```

## Training Data

- **Total tweets**: 8,000
- **Positive class**: 4,000 (50%)
- **Negative class**: 4,000 (50%)
- **Split**: 80% train (6,400), 20% test (1,600)
- **Source**: NLTK twitter_samples corpus

## Generated Artifacts

### Models Directory (`preprocessed_data/models/`)
```
models/
├── sklearn_logistic_model.pkl      # Trained sklearn model
├── custom_logistic_model.pkl       # Trained custom model
├── model_metadata.json             # Configuration and metrics
└── training.log                    # Complete training logs
```

### Visualizations Directory (`preprocessed_data/visualizations/`)
```
visualizations/
├── decision_boundary_sklearn.png   # Scatter plot + decision line
├── decision_boundary_custom.png    # Custom model decision boundary
├── roc_curves.png                  # ROC-AUC comparison
└── training_loss.png               # Custom model loss curve
```

## Visualization Guide

### Decision Boundary Plot
Shows:
- **Red dots**: Negative sentiment tweets
- **Green dots**: Positive sentiment tweets
- **Blue line**: Linear decision boundary (z=0)
- **Green arrow**: Direction of positive prediction
- **Red arrow**: Direction of negative prediction
- **Dashed contours**: Prediction confidence (0.5 probability line)

### ROC Curves
- **X-axis**: False Positive Rate (1 - Specificity)
- **Y-axis**: True Positive Rate (Sensitivity/Recall)
- **Diagonal**: Random classifier baseline
- **AUC**: Area Under Curve (higher = better)
  - SKlearn: AUC = 1.0000 (perfect discrimination)
  - Custom: AUC = 1.0000 (perfect discrimination)

### Training Loss
- **Y-axis**: Binary cross-entropy loss
- **X-axis**: Gradient descent iteration
- Shows convergence of custom model over 5000 iterations
- Loss decreases from ~0.65 to ~0.22

## Usage Examples

### Basic Prediction
```python
from model_inference import LogisticRegressionInference

inference = LogisticRegressionInference()

# Single prediction
result = inference.predict("I love this product!", model='sklearn')
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")
```

### Batch Processing
```python
tweets = [
    "Amazing experience!",
    "I hate this",
    "Not bad"
]

results = inference.batch_predict(tweets, model='sklearn')
for result in results:
    print(f"{result['tweet']:20s} -> {result['sentiment']}")
```

### Model Interpretation
```python
coefs = inference.get_model_coefficients()
print(f"Positive words coefficient: {coefs['positive_words_coef']:.6f}")
print(f"Negative words coefficient: {coefs['negative_words_coef']:.6f}")
```

## Implementation Details

### Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))

Properties:
- σ(0) = 0.5
- lim(z→∞) σ(z) = 1
- lim(z→-∞) σ(z) = 0
- Derivative: σ'(z) = σ(z)(1 - σ(z))
```

### Probability Calibration
```
P(y=1|x) = σ(θ₀ + θ₁*x₁ + θ₂*x₂)

For > 50% probability → predict positive
For ≤ 50% probability → predict negative
```

### Gradient Descent Step
```
θ := θ - α * ∇J(θ)

Where:
α   = learning rate (0.001)
∇J  = gradient of cost function
θ   = model parameters
```

### L2 Regularization
```
Penalty term: λ/(2m) * Σ θ²[1:]

Effect:
- Prevents overfitting
- Shrinks weights toward zero
- λ = 0.01 (weak regularization)
```

## Configuration Parameters

See `config.py` and `logistic_regression_model.py` for:
- Feature extraction settings
- Model hyperparameters
- Train/test split ratios
- Cross-validation folds
- Regularization strength

## Performance Metrics Summary

| Metric | SKlearn | Custom | Interpretation |
|--------|---------|--------|-----------------|
| Accuracy | 99.69% | 96.13% | % correct predictions |
| Precision | 99.38% | 92.81% | % predicted positive that are |
| Recall | 100% | 100% | % actual positive found |
| F1-Score | 99.69% | 96.27% | Harmonic mean |
| ROC-AUC | 100% | 100% | Discrimination ability |

## Advanced Features

### Decision Confidence Contours
- 0.95 probability: Very confident positive
- 0.5 probability: Decision boundary
- 0.05 probability: Very confident negative

### Feature Scaling
- Z-score normalization applied to semantic features
- Bias term remains at 1
- Improves gradient descent convergence

### Cross-Validation
- Stratified K-Fold with k=5
- Maintains class balance across folds
- Reduces variance in performance estimates

## Troubleshooting

### Issue: Low recall on positive class
→ Decrease threshold from 0.5 to 0.4

### Issue: Too many false positives
→ Increase threshold or use custom model

### Issue: Model not converging
→ Reduce learning rate or increase iterations

## Future Enhancements

1. **Multi-class sentiment** (positive/neutral/negative)
2. **Word importance analysis** via SHAP values
3. **Confidence calibration** (Platt scaling)
4. **Threshold optimization** for different use cases
5. **Feature engineering** (n-grams, TF-IDF weights)
6. **Ensemble methods** (stacking, voting)
7. **Deep learning** (LSTM, BERT)

## References

- Binary logistic regression theory
- Gradient descent optimization
- Sigmoid activation function
- Cross-entropy loss function
- ROC-AUC evaluation metrics
- NLTK twitter_samples corpus

## File Structure

```
/Users/vardaankapoor/Documents/NLP/
├── logistic_regression_model.py     # Main pipeline
├── model_inference.py               # Inference interface
├── run_complete_pipeline.py         # End-to-end orchestrator
├── tweet_preprocessing.py           # Preprocessing module
├── config.py                        # Configuration
├── requirements.txt                 # Dependencies
└── preprocessed_data/
    ├── models/
    │   ├── sklearn_logistic_model.pkl
    │   ├── custom_logistic_model.pkl
    │   ├── model_metadata.json
    │   └── training.log
    └── visualizations/
        ├── decision_boundary_sklearn.png
        ├── decision_boundary_custom.png
        ├── roc_curves.png
        └── training_loss.png
```

## Running the Pipeline

```bash
# Complete pipeline
python run_complete_pipeline.py

# Just logistic regression
python logistic_regression_model.py

# Interactive inference
python model_inference.py
```

## Key Achievements

✓ **99.69% accuracy** on test set (sklearn model)
✓ **100% ROC-AUC** - perfect discrimination between classes
✓ **Perfect recall** on positive class (100%)
✓ **Dual implementation** - both sklearn and custom gradient descent
✓ **Professional visualizations** - decision boundaries, ROC curves, loss plots
✓ **Comprehensive evaluation** - stratified cross-validation, confusion matrices
✓ **Production-ready** - model serialization, metadata, error handling
✓ **Interactive inference** - command-line prediction interface

"""
