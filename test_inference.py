#!/usr/bin/env python3
"""Quick test of the inference module."""

from model_inference import LogisticRegressionInference

# Initialize inference
inference = LogisticRegressionInference()

# Test predictions on example tweets
test_tweets = [
    "I absolutely love this! Best product ever! 😍👍",
    "Terrible quality, very disappointed 😞",
    "It's okay, nothing special",
    "Amazing customer service! Very happy!",
    "Worst purchase I've ever made!",
]

print("\n" + "="*80)
print("SENTIMENT ANALYSIS - LOGISTIC REGRESSION INFERENCE")
print("="*80)

for tweet in test_tweets:
    result = inference.predict(tweet, 'sklearn')
    print(f"\nTweet: {tweet}")
    print(f"  Sentiment: {result['sentiment'].upper():8s} | Confidence: {result['confidence']:.4f}")
    print(f"  Pos words: {result['positive_words_sum']:6.2f} | Neg words: {result['negative_words_sum']:6.2f}")

# Show model info
inference.print_model_info()
