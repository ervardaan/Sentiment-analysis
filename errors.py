#!/usr/bin/env python3
"""
errors.py  –  Error Analysis Module (Coursera Week 1 Part 5)
=============================================================

Analyses every misclassified tweet in the test set and surfaces:
  * the raw tweet
  * the processed token list
  * the model's predicted probability
  * the true label
  * a brief reason category (overly positive/negative words, proper nouns, …)

Usage (standalone):
    python errors.py

Usage (API):
    from errors import ErrorAnalyser
    ea = ErrorAnalyser(test_x, test_y, freqs, theta)
    report = ea.run()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import process_tweet
from sentiment_model import predict_tweet, extract_features

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        format="%(asctime)s  [%(levelname)s]  %(message)s", level=logging.INFO
    )


# ─────────────────────────────────────────────────────────────────── #
#  Data classes                                                       #
# ─────────────────────────────────────────────────────────────────── #
@dataclass
class MisclassifiedTweet:
    index:         int
    raw_tweet:     str
    processed:     List[str]
    true_label:    int          # 1 = positive, 0 = negative
    predicted_prob:float
    predicted_label:int
    pos_freq_sum:  float
    neg_freq_sum:  float
    reason:        str = ""


@dataclass
class ErrorReport:
    total_test:      int = 0
    total_errors:    int = 0
    accuracy:        float = 0.0
    false_positives: List[MisclassifiedTweet] = field(default_factory=list)
    false_negatives: List[MisclassifiedTweet] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────── #
#  Reason heuristics                                                  #
# ─────────────────────────────────────────────────────────────────── #
_CONTRACTIONS = {
    "happi": "contains happy variant",
    "sad":   "contains sad",
    "not":   "negation present",
    "no":    "negation present",
    "love":  "positive word",
    "hate":  "negative word",
}

def _guess_reason(processed: List[str], true_label: int) -> str:
    """Heuristic explanation of why the model got it wrong."""
    tokens = set(processed)
    if true_label == 1:      # should be positive, predicted negative
        neg_words = [t for t in tokens if t in {"sad", "hate", "bad", "wrong", "miss"}]
        if neg_words:
            return f"Contains strong negative words: {neg_words}"
        if any(t.startswith(":") or t.startswith(";") for t in tokens):
            return "Emoticon ambiguity"
        return "Low positive signal – rare or unseen positive words"
    else:                     # should be negative, predicted positive
        pos_words = [t for t in tokens if t in {"happi", "love", "great", "good", "amaz"}]
        if pos_words:
            return f"Contains strong positive words: {pos_words}"
        if any(t.startswith(":") or t.startswith(";") for t in tokens):
            return "Emoticon ambiguity"
        return "Low negative signal – rare or unseen negative words"


# ─────────────────────────────────────────────────────────────────── #
#  Main analyser class                                               #
# ─────────────────────────────────────────────────────────────────── #
class ErrorAnalyser:
    """Identifies and explains every misclassified tweet.

    Parameters
    ----------
    test_x : list[str]
        Raw test tweets (2 000).
    test_y : np.ndarray
        Ground-truth labels, shape (2000, 1).
    freqs  : dict
        Frequency dictionary from build_freqs.
    theta  : np.ndarray
        Trained weight vector, shape (3, 1).
    threshold : float
        Decision boundary (default 0.5).
    """

    def __init__(
        self,
        test_x: List[str],
        test_y: np.ndarray,
        freqs: Dict[Tuple[str, float], int],
        theta: np.ndarray,
        threshold: float = 0.5,
    ) -> None:
        self.test_x    = test_x
        self.test_y    = test_y
        self.freqs     = freqs
        self.theta     = theta
        self.threshold = threshold

    # ────────────────────────────────────────────────── #
    def run(self) -> ErrorReport:
        """Scan all test tweets, flag misclassified ones, return report."""
        report = ErrorReport(total_test=len(self.test_x))

        y_true = np.squeeze(self.test_y).astype(int)

        for idx, (tweet, true_lbl) in enumerate(zip(self.test_x, y_true)):
            prob  = float(predict_tweet(tweet, self.freqs, self.theta))
            pred  = int(prob >= self.threshold)

            if pred != true_lbl:
                processed = process_tweet(tweet)
                feat      = extract_features(tweet, self.freqs)
                pos_sum   = float(feat[0, 1])
                neg_sum   = float(feat[0, 2])
                reason    = _guess_reason(processed, true_lbl)

                mc = MisclassifiedTweet(
                    index=idx,
                    raw_tweet=tweet,
                    processed=processed,
                    true_label=int(true_lbl),
                    predicted_prob=round(prob, 6),
                    predicted_label=pred,
                    pos_freq_sum=pos_sum,
                    neg_freq_sum=neg_sum,
                    reason=reason,
                )
                if true_lbl == 1:
                    report.false_negatives.append(mc)  # positive but called negative
                else:
                    report.false_positives.append(mc)  # negative but called positive

        report.total_errors  = len(report.false_positives) + len(report.false_negatives)
        report.accuracy      = 1.0 - report.total_errors / max(report.total_test, 1)
        return report

    # ────────────────────────────────────────────────── #
    #  Pretty-print helpers                              #
    # ────────────────────────────────────────────────── #
    @staticmethod
    def print_report(report: ErrorReport, max_show: int = 20) -> None:
        sep = "─" * 70
        print(f"\n{'═'*70}")
        print("  ERROR ANALYSIS REPORT")
        print(f"{'═'*70}")
        print(f"  Test set size : {report.total_test}")
        print(f"  Misclassified : {report.total_errors}")
        print(f"  Accuracy      : {report.accuracy:.4f} ({report.accuracy*100:.2f}%)")
        print(f"  False positives (negative tweets predicted positive): {len(report.false_positives)}")
        print(f"  False negatives (positive tweets predicted negative): {len(report.false_negatives)}")

        def _show(items: List[MisclassifiedTweet], title: str) -> None:
            print(f"\n{sep}")
            print(f"  {title}")
            print(sep)
            for mc in items[:max_show]:
                print(f"\n  [{mc.index:4d}] TRUE={mc.true_label}  PRED={mc.predicted_label}  "
                      f"PROB={mc.predicted_prob:.4f}")
                print(f"        POS_SUM={mc.pos_freq_sum:.0f}  NEG_SUM={mc.neg_freq_sum:.0f}")
                print(f"        RAW:       {mc.raw_tweet}")
                print(f"        PROCESSED: {mc.processed}")
                print(f"        REASON:    {mc.reason}")
            if len(items) > max_show:
                print(f"\n  … {len(items) - max_show} more (see saved JSON)")

        _show(report.false_positives, "FALSE POSITIVES  (negative → predicted positive)")
        _show(report.false_negatives, "FALSE NEGATIVES  (positive → predicted negative)")
        print(f"\n{'═'*70}\n")

    @staticmethod
    def save_report(report: ErrorReport, path: Path) -> None:
        """Serialise the report as JSON for downstream use."""
        def mc_to_dict(mc: MisclassifiedTweet) -> dict:
            return {
                "index":          mc.index,
                "raw_tweet":      mc.raw_tweet,
                "processed":      mc.processed,
                "true_label":     mc.true_label,
                "predicted_prob": mc.predicted_prob,
                "predicted_label":mc.predicted_label,
                "pos_freq_sum":   mc.pos_freq_sum,
                "neg_freq_sum":   mc.neg_freq_sum,
                "reason":         mc.reason,
            }
        data = {
            "total_test":      report.total_test,
            "total_errors":    report.total_errors,
            "accuracy":        report.accuracy,
            "false_positives": [mc_to_dict(m) for m in report.false_positives],
            "false_negatives": [mc_to_dict(m) for m in report.false_negatives],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Error report saved → {path}")


# ─────────────────────────────────────────────────────────────────── #
#  Standalone CLI                                                     #
# ─────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    import pickle
    from pathlib import Path

    MODELS = Path("preprocessed_data/models")

    # load artefacts produced by trainer.py
    with open(MODELS / "freqs.pkl", "rb") as f:
        freqs = pickle.load(f)
    theta = np.load(MODELS / "theta_gd.npy")

    # reload test split
    import nltk
    nltk.download("twitter_samples", quiet=True)
    from nltk.corpus import twitter_samples
    pos = twitter_samples.strings("positive_tweets.json")
    neg = twitter_samples.strings("negative_tweets.json")
    test_x = pos[4000:] + neg[4000:]
    test_y = np.append(np.ones((1000, 1)), np.zeros((1000, 1)), axis=0)

    ea = ErrorAnalyser(test_x, test_y, freqs, theta)
    report = ea.run()
    ErrorAnalyser.print_report(report)
    ErrorAnalyser.save_report(report, MODELS / "error_report.json")
