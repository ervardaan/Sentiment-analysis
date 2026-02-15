#!/usr/bin/env python3
"""Single-file tweet preprocessing and vectorization pipeline.

This script merges preprocessing, vectorization, and corpus orchestration
into a single, self-contained module. It loads the NLTK twitter_samples
corpus, preprocesses all tweets, fits a TF-IDF vectorizer, and saves
results to the `preprocessed_data/` directory.

Run:
    python tweet_preprocessing.py
"""

from __future__ import annotations

import os
import re
import json
import string
import logging
import pickle
from typing import List, Tuple, Dict
from datetime import datetime

import nltk
from nltk.corpus import twitter_samples, stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from tqdm import tqdm


# Configuration (kept local for single-file simplicity)
OUTPUT_DIR = "preprocessed_data"
VECTORS_FILE = os.path.join(OUTPUT_DIR, "tweet_vectors.pkl")
TOKENS_FILE = os.path.join(OUTPUT_DIR, "tweet_tokens.json")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")
VECTORIZER_FILE = os.path.join(OUTPUT_DIR, "vectorizer.pkl")
FEATURE_NAMES_FILE = os.path.join(OUTPUT_DIR, "feature_names.json")
ORIGINAL_TWEETS_FILE = os.path.join(OUTPUT_DIR, "original_tweets.json")

PRESERVE_CASE = False
STRIP_HANDLES = True
REDUCE_LEN = True
STOPWORDS_LANG = "english"

VECTORIZER_MAX_FEATURES = 5000
VECTORIZER_MIN_DF = 2
VECTORIZER_MAX_DF = 0.8
VECTORIZER_NGRAM_RANGE = (1, 2)


class TweetPreprocessor:
    """Combined preprocessor used by the single-file pipeline."""

    def __init__(self,
                 preserve_case: bool = PRESERVE_CASE,
                 strip_handles: bool = STRIP_HANDLES,
                 reduce_len: bool = REDUCE_LEN,
                 stopwords_lang: str = STOPWORDS_LANG):
        self.tokenizer = TweetTokenizer(preserve_case=preserve_case,
                                        strip_handles=strip_handles,
                                        reduce_len=reduce_len)
        self.stemmer = PorterStemmer()
        self.stopwords_set = set(stopwords.words(stopwords_lang))

    def _remove_retweet_markers(self, tweet: str) -> str:
        return re.sub(r"^RT[\s]+", "", tweet)

    def _remove_hyperlinks(self, tweet: str) -> str:
        return re.sub(r"https?://[^\s\n\r]+", "", tweet)

    def _remove_hash_symbols(self, tweet: str) -> str:
        return re.sub(r"#", "", tweet)

    def _remove_extra_whitespace(self, tweet: str) -> str:
        return re.sub(r"\s+", " ", tweet).strip()

    def process(self, tweet: str) -> List[str]:
        tweet = self._remove_retweet_markers(tweet)
        tweet = self._remove_hyperlinks(tweet)
        tweet = self._remove_hash_symbols(tweet)
        tweet = self._remove_extra_whitespace(tweet)

        tokens = self.tokenizer.tokenize(tweet)
        filtered = [w for w in tokens if w not in self.stopwords_set and w not in string.punctuation]
        stemmed = [self.stemmer.stem(w) for w in filtered]
        return stemmed


class TweetVectorizer:
    """TF-IDF vectorizer encapsulation used by the single-file pipeline."""

    def __init__(self, max_features=VECTORIZER_MAX_FEATURES,
                 min_df=VECTORIZER_MIN_DF, max_df=VECTORIZER_MAX_DF,
                 ngram_range=VECTORIZER_NGRAM_RANGE):
        self.vectorizer = TfidfVectorizer(max_features=max_features,
                          min_df=min_df,
                          max_df=max_df,
                          ngram_range=ngram_range,
                          analyzer='word',
                          lowercase=True,
                          token_pattern=r'(?u)\b\w+\b')
        self.is_fitted = False

    def fit_transform(self, documents: List[str]) -> Tuple[csr_matrix, Dict]:
        vectors = self.vectorizer.fit_transform(documents)
        self.is_fitted = True
        metadata = {
            "vector_shape": vectors.shape,
            "vocab_size": len(self.vectorizer.get_feature_names_out()),
            "max_features": self.vectorizer.max_features
        }
        return vectors, metadata

    def transform(self, documents: List[str]) -> csr_matrix:
        if not self.is_fitted:
            raise RuntimeError("Vectorizer not fitted")
        return self.vectorizer.transform(documents)


def ensure_nltk_data():
    for pkg in ("twitter_samples", "stopwords"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            print(f"Downloading NLTK resource: {pkg}")
            nltk.download(pkg)


def setup_logger() -> logging.Logger:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = logging.getLogger("tweet_pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(OUTPUT_DIR, "processing_log.txt"))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def run_pipeline():
    logger = setup_logger()
    logger.info("Starting tweet preprocessing pipeline")

    ensure_nltk_data()

    # Load corpus
    pos = twitter_samples.strings("positive_tweets.json")
    neg = twitter_samples.strings("negative_tweets.json")
    all_tweets = pos + neg
    labels = ["positive"] * len(pos) + ["negative"] * len(neg)

    logger.info(f"Loaded {len(all_tweets)} tweets")

    pre = TweetPreprocessor()
    processed = []
    total_before = 0
    total_after = 0

    for tweet in tqdm(all_tweets, desc="Preprocessing tweets"):
        tokens = pre.process(tweet)
        processed.append(tokens)
        total_before += len(tweet.split())
        total_after += len(tokens)

    avg_before = total_before / len(all_tweets)
    avg_after = total_after / len(all_tweets)
    logger.info(f"Avg tokens before: {avg_before:.2f}, after: {avg_after:.2f}")

    # Vectorize
    docs = [" ".join(toks) for toks in processed]
    vec = TweetVectorizer()
    vectors, vec_meta = vec.fit_transform(docs)
    logger.info(f"Vector shape: {vec_meta['vector_shape']}, vocab_size: {vec_meta['vocab_size']}")

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(VECTORS_FILE, 'wb') as f:
        pickle.dump(vectors, f)
    with open(TOKENS_FILE, 'w') as f:
        json.dump(processed, f)
    with open(ORIGINAL_TWEETS_FILE, 'w') as f:
        json.dump([{"id": i, "text": t, "label": l} for i, (t, l) in enumerate(zip(all_tweets, labels))], f)
    with open(FEATURE_NAMES_FILE, 'w') as f:
        json.dump(list(vec.vectorizer.get_feature_names_out()), f)
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vec.vectorizer, f)
    meta = {
        "total_tweets": len(all_tweets),
        "avg_tokens_before": avg_before,
        "avg_tokens_after": avg_after,
        "vector_meta": vec_meta,
        "timestamp": datetime.now().isoformat()
    }
    with open(METADATA_FILE, 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info("All outputs saved to preprocessed_data/")


def main():
    try:
        run_pipeline()
        print("Processing complete. Data saved to preprocessed_data/")
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == '__main__':
    main()

