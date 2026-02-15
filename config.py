"""Configuration for tweet preprocessing pipeline."""

import os

# Preprocessing settings
PRESERVE_CASE = False
STRIP_HANDLES = True
REDUCE_LEN = True

# Stopwords language
STOPWORDS_LANG = "english"

# Vectorization settings
VECTORIZER_MAX_FEATURES = 5000
VECTORIZER_MIN_DF = 2
VECTORIZER_MAX_DF = 0.8
VECTORIZER_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams

# Output paths
OUTPUT_DIR = "preprocessed_data"
VECTORS_FILE = os.path.join(OUTPUT_DIR, "tweet_vectors.pkl")
TOKENS_FILE = os.path.join(OUTPUT_DIR, "tweet_tokens.json")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")
VECTORIZER_FILE = os.path.join(OUTPUT_DIR, "vectorizer.pkl")
FEATURE_NAMES_FILE = os.path.join(OUTPUT_DIR, "feature_names.json")
ORIGINAL_TWEETS_FILE = os.path.join(OUTPUT_DIR, "original_tweets.json")

# Logging
LOG_FILE = os.path.join(OUTPUT_DIR, "processing_log.txt")
LOG_LEVEL = "INFO"
