#!/usr/bin/env python3
"""
utils.py  –  NLP utility functions for the Coursera LR assignment
==================================================================

Provides:
    process_tweet(tweet)          → List[str]
    build_freqs(tweets, ys)       → Dict[(str,float), int]

Both functions are identical in contract to the Coursera reference versions
so that any code importing `from utils import process_tweet, build_freqs`
works out of the box.
"""

from __future__ import annotations

import re
import string
from typing import Dict, List, Tuple

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


# --------------------------------------------------------------------------- #
#  Lazy-loaded singletons (initialised once to avoid recreation per call)      #
# --------------------------------------------------------------------------- #

_tokenizer: TweetTokenizer | None = None
_stemmer: PorterStemmer | None = None
_stop_words: frozenset | None = None


def _get_tokenizer() -> TweetTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = TweetTokenizer(
            preserve_case=False,
            strip_handles=True,
            reduce_len=True,
        )
    return _tokenizer


def _get_stemmer() -> PorterStemmer:
    global _stemmer
    if _stemmer is None:
        _stemmer = PorterStemmer()
    return _stemmer


def _get_stop_words() -> frozenset:
    global _stop_words
    if _stop_words is None:
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        _stop_words = frozenset(stopwords.words("english"))
    return _stop_words


# --------------------------------------------------------------------------- #
#  Core NLP functions                                                          #
# --------------------------------------------------------------------------- #

def process_tweet(tweet: str) -> List[str]:
    """Clean, tokenise, remove stop-words, and stem a single tweet.

    Pipeline (matches Coursera week-1 reference exactly):
      1. Remove "RT …" retweet markers
      2. Remove hyperlinks (http / https)
      3. Strip '#' from hashtags (keep the word)
      4. Lowercase + tokenise with TweetTokenizer
         (strip_handles=True, reduce_len=True)
      5. Remove English stop-words and bare punctuation tokens
      6. Porter-stem every surviving token

    Args:
        tweet: Raw tweet string.

    Returns:
        List of stemmed, cleaned tokens.  May be empty.

    Examples:
        >>> process_tweet("#FollowFriday @France_Inte for being top engaged members :)")
        ['followfriday', 'top', 'engag', 'member', ':)']
    """
    tokenizer = _get_tokenizer()
    stemmer = _get_stemmer()
    stop_words = _get_stop_words()

    # 1 – remove retweet marker
    tweet = re.sub(r"^RT[\s]+", "", tweet)

    # 2 – remove URLs
    tweet = re.sub(r"https?://[^\s\n\r]+", "", tweet)

    # 3 – strip hash symbol (keep text)
    tweet = re.sub(r"#", "", tweet)

    # 4 – tokenise
    tokens: List[str] = tokenizer.tokenize(tweet)

    # 5 – remove stop-words and plain punctuation
    punct = set(string.punctuation)
    cleaned: List[str] = [
        tok for tok in tokens
        if tok not in stop_words and tok not in punct
    ]

    # 6 – stem
    stemmed: List[str] = [stemmer.stem(tok) for tok in cleaned]

    return stemmed


def build_freqs(tweets: List[str], ys) -> Dict[Tuple[str, float], int]:
    """Build a frequency dictionary mapping (word, label) → count.

    Iterates over every tweet, processes it, and increments a counter for
    each (word, label) pair encountered.

    Args:
        tweets: Iterable of raw tweet strings (length m).
        ys:     Labels array/list of shape (m,) or (m, 1).
                Values are expected to be 0.0 (negative) or 1.0 (positive).

    Returns:
        freqs: ``{(word, label): count}``

    Example:
        >>> freqs = build_freqs(train_x, train_y)
        >>> freqs.get((':)', 1.0), 0)   # how often ':)' appears in positives
        3568
    """
    # Flatten labels to a plain Python list of floats
    ys_flat: List[float] = np.squeeze(np.asarray(ys, dtype=float)).tolist()

    freqs: Dict[Tuple[str, float], int] = {}

    for y, tweet in zip(ys_flat, tweets):
        for word in process_tweet(tweet):
            pair: Tuple[str, float] = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1

    return freqs
