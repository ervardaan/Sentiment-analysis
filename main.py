#!/usr/bin/env python3
"""
main.py  –  Application entry-point
=====================================

Ties together every stage of the SDLC and ML lifecycle:

    1. Train / evaluate all models          (trainer.py)
    2. Run error analysis on test set       (errors.py)
    3. Interactive inference loop (optional)(model_inference.py)

Usage:
    python main.py                    # full pipeline
    python main.py --skip-inference   # skip interactive loop
    python main.py --errors-only      # load saved artefacts, only error analysis
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


MODELS_DIR = Path("preprocessed_data/models")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sentiment analysis – full pipeline")
    p.add_argument("--skip-inference", action="store_true",
                   help="Skip interactive tweet prediction loop")
    p.add_argument("--errors-only", action="store_true",
                   help="Load saved artefacts and only run error analysis")
    return p.parse_args()


def _run_training() -> dict:
    from trainer import SentimentTrainer
    t = SentimentTrainer()
    return t.run()


def _run_errors(artefacts: dict) -> None:
    from errors import ErrorAnalyser
    ea = ErrorAnalyser(
        test_x=artefacts["test_x"],
        test_y=artefacts["test_y"],
        freqs=artefacts["freqs"],
        theta=artefacts["theta_gd"],
    )
    report = ea.run()
    ErrorAnalyser.print_report(report)
    ErrorAnalyser.save_report(report, MODELS_DIR / "error_report.json")


def _run_inference(artefacts: dict) -> None:
    from errors import ErrorAnalyser
    freqs = artefacts["freqs"]
    theta = artefacts["theta_gd"]

    print("\n" + "=" * 60)
    print("  Interactive sentiment prediction")
    print("  Type a tweet and press Enter.  Type 'quit' to exit.")
    print("=" * 60)

    from sentiment_model import predict_tweet
    while True:
        try:
            tweet = input("\nTweet > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if tweet.lower() in ("quit", "exit", "q"):
            break
        if not tweet:
            continue
        prob = float(predict_tweet(tweet, freqs, theta))
        sentiment = "POSITIVE 😊" if prob >= 0.5 else "NEGATIVE 😞"
        print(f"  Probability : {prob:.6f}")
        print(f"  Sentiment   : {sentiment}")


def _load_saved_artefacts() -> dict:
    """Load artefacts saved by trainer.py (for --errors-only mode)."""
    import nltk
    nltk.download("twitter_samples", quiet=True)
    from nltk.corpus import twitter_samples
    from sentiment_model import extract_features_batch

    with open(MODELS_DIR / "freqs.pkl", "rb") as f:
        freqs = pickle.load(f)
    theta_gd = np.load(MODELS_DIR / "theta_gd.npy")

    pos = twitter_samples.strings("positive_tweets.json")
    neg = twitter_samples.strings("negative_tweets.json")
    test_x = pos[4000:] + neg[4000:]
    test_y = np.append(np.ones((1000, 1)), np.zeros((1000, 1)), axis=0)

    return {
        "freqs":   freqs,
        "theta_gd": theta_gd,
        "test_x":  test_x,
        "test_y":  test_y,
    }


def main() -> None:
    args = _parse_args()

    if args.errors_only:
        print("Loading saved artefacts …")
        artefacts = _load_saved_artefacts()
    else:
        artefacts = _run_training()

    _run_errors(artefacts)

    if not args.skip_inference:
        _run_inference(artefacts)


if __name__ == "__main__":
    main()
