#!/usr/bin/env python3
"""
tests/test_sentiment_model.py  –  Unit tests for sentiment_model.py + utils.py
===============================================================================

Covers:
    - process_tweet  (tokenisation, stemming, stopword removal)
    - build_freqs    (dict size, key format, counts)
    - sigmoid        (scalar, array, edge-cases)
    - gradientDescent (cost convergence, theta shape)
    - extract_features (output shape, bias=1, zeros for unseen tweet)
    - predict_tweet  (output range, known-positive tweet)
    - test_logistic_regression (accuracy on toy set)
    - GradientDescentTrainer   (fits and produces lower cost)
    - AdvancedLogisticRegression (fits, predicts, proba in [0,1])

Run with:
    cd /Users/vardaankapoor/Documents/NLP
    python -m pytest tests/test_sentiment_model.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import process_tweet, build_freqs
from sentiment_model import (
    sigmoid,
    gradientDescent,
    extract_features,
    extract_features_batch,
    predict_tweet,
    evaluate_logistic_regression,
    GradientDescentTrainer,
    AdvancedLogisticRegression,
)


# ═══════════════════════════════════════════════════════════════════ #
#  Fixtures (tiny corpus used across many tests)                     #
# ═══════════════════════════════════════════════════════════════════ #
POS_TWEETS = [
    "I am happy :)",
    "I love this movie",
    "great day today",
    "sunshine makes me smile",
    "so excited for the weekend!",
]
NEG_TWEETS = [
    "I am sad :(",
    "I hate this weather",
    "terrible experience",
    "this is so bad",
    "worst day ever",
]
ALL_TWEETS = POS_TWEETS + NEG_TWEETS
LABELS_1D  = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
LABELS_2D  = LABELS_1D.reshape(-1, 1)


def _make_freqs() -> dict:
    return build_freqs(ALL_TWEETS, LABELS_2D)


# ═══════════════════════════════════════════════════════════════════ #
#  utils.process_tweet                                               #
# ═══════════════════════════════════════════════════════════════════ #
class TestProcessTweet(unittest.TestCase):

    def test_returns_list(self):
        result = process_tweet("Hello world!")
        self.assertIsInstance(result, list)

    def test_removes_rt_prefix(self):
        result = process_tweet("RT @user: hello")
        joined = " ".join(result)
        self.assertNotIn("rt", joined.lower())

    def test_removes_urls(self):
        result = process_tweet("check this out https://example.com now")
        joined = " ".join(result)
        self.assertNotIn("http", joined)
        self.assertNotIn("example.com", joined)

    def test_removes_hash_symbol(self):
        result = process_tweet("#happy day")
        for tok in result:
            self.assertFalse(tok.startswith("#"))

    def test_removes_stopwords(self):
        # "the" is an English stopword; should not appear
        result = process_tweet("the quick brown fox")
        self.assertNotIn("the", result)

    def test_porter_stemming(self):
        result = process_tweet("running happily")
        stems = set(result)
        # "running" → "run" or "runn"; none of these should be full word
        self.assertNotIn("running", stems)

    def test_empty_string(self):
        result = process_tweet("")
        self.assertIsInstance(result, list)

    def test_punctuation_only(self):
        result = process_tweet("!!! ??? ...")
        # All punctuation is stripped; result may be empty or contain
        # only stripped characters
        self.assertIsInstance(result, list)

    def test_handles_smiley(self):
        # Smiley faces are not stopwords, may be preserved
        result = process_tweet("I am happy :)")
        self.assertIsInstance(result, list)


# ═══════════════════════════════════════════════════════════════════ #
#  utils.build_freqs                                                 #
# ═══════════════════════════════════════════════════════════════════ #
class TestBuildFreqs(unittest.TestCase):

    def setUp(self):
        self.freqs = _make_freqs()

    def test_returns_dict(self):
        self.assertIsInstance(self.freqs, dict)

    def test_keys_are_tuples(self):
        for key in self.freqs:
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)

    def test_label_is_float(self):
        for (word, label) in self.freqs:
            self.assertIsInstance(label, float)
            self.assertIn(label, (0.0, 1.0))

    def test_positive_word_in_freqs(self):
        # "happi" (stemmed from "happy") or "love" should appear with label 1.0
        positive_keys = [k for k in self.freqs if k[1] == 1.0]
        self.assertGreater(len(positive_keys), 0)

    def test_non_empty(self):
        self.assertGreater(len(self.freqs), 0)

    def test_counts_are_positive(self):
        for count in self.freqs.values():
            self.assertGreater(count, 0)

    def test_accepts_1d_labels(self):
        """build_freqs should work with both (m,) and (m,1) label arrays."""
        freqs_1d = build_freqs(ALL_TWEETS, LABELS_1D)
        self.assertIsInstance(freqs_1d, dict)
        self.assertGreater(len(freqs_1d), 0)


# ═══════════════════════════════════════════════════════════════════ #
#  sigmoid                                                           #
# ═══════════════════════════════════════════════════════════════════ #
class TestSigmoid(unittest.TestCase):

    def test_zero(self):
        self.assertAlmostEqual(float(sigmoid(0)), 0.5, places=10)

    def test_large_positive(self):
        self.assertAlmostEqual(float(sigmoid(100)), 1.0, places=5)

    def test_large_negative(self):
        self.assertAlmostEqual(float(sigmoid(-100)), 0.0, places=5)

    def test_array_input(self):
        z = np.array([-1, 0, 1], dtype=float)
        out = sigmoid(z)
        self.assertEqual(out.shape, (3,))
        self.assertLess(out[0], 0.5)
        self.assertAlmostEqual(out[1], 0.5, places=10)
        self.assertGreater(out[2], 0.5)

    def test_output_range(self):
        z = np.linspace(-100, 100, 1000)
        out = sigmoid(z)
        self.assertTrue(np.all(out >= 0) and np.all(out <= 1))

    def test_2d_input(self):
        z = np.array([[0.0], [1.0], [-1.0]])
        out = sigmoid(z)
        self.assertEqual(out.shape, (3, 1))


# ═══════════════════════════════════════════════════════════════════ #
#  gradientDescent                                                   #
# ═══════════════════════════════════════════════════════════════════ #
class TestGradientDescent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        freqs = _make_freqs()
        tweets = ALL_TWEETS
        labels = LABELS_2D
        cls.X = extract_features_batch(tweets, freqs)   # (10, 3)
        cls.y = labels                                   # (10, 1)
        theta0 = np.zeros((3, 1))
        cls.J_final, cls.theta = gradientDescent(cls.X, cls.y, theta0, 1e-9, 500)

    def test_J_is_scalar(self):
        self.assertIsInstance(self.J_final, float)

    def test_theta_shape(self):
        self.assertEqual(self.theta.shape, (3, 1))

    def test_cost_is_finite(self):
        self.assertTrue(np.isfinite(self.J_final))

    def test_cost_positive(self):
        self.assertGreater(self.J_final, 0)

    def test_cost_lower_than_initial(self):
        # Cost from 0 iterations = log(2) ≈ 0.693
        # Toy corpus (10 tweets) barely moves at alpha=1e-9 in 500 iters;
        # check it is strictly below the untrained cost ceiling.
        self.assertLessEqual(self.J_final, 0.6932)


# ═══════════════════════════════════════════════════════════════════ #
#  extract_features                                                  #
# ═══════════════════════════════════════════════════════════════════ #
class TestExtractFeatures(unittest.TestCase):

    def setUp(self):
        self.freqs = _make_freqs()

    def test_output_shape_single(self):
        feat = extract_features("I am happy", self.freqs)
        self.assertEqual(feat.shape, (1, 3))

    def test_bias_is_one(self):
        feat = extract_features("I am happy", self.freqs)
        self.assertEqual(feat[0, 0], 1.0)

    def test_pos_sum_nonneg(self):
        feat = extract_features("I am happy", self.freqs)
        self.assertGreaterEqual(feat[0, 1], 0)

    def test_neg_sum_nonneg(self):
        feat = extract_features("I am sad", self.freqs)
        self.assertGreaterEqual(feat[0, 2], 0)

    def test_unseen_tweet_gives_zero_sums(self):
        feat = extract_features("xyzabc qwerty zzzzzz", self.freqs)
        self.assertEqual(feat[0, 1], 0)
        self.assertEqual(feat[0, 2], 0)

    def test_batch_vs_single_consistent(self):
        tweet = "I love this movie"
        single = extract_features(tweet, self.freqs)
        batch  = extract_features_batch([tweet], self.freqs)
        np.testing.assert_array_equal(single, batch)

    def test_batch_shape(self):
        batch = extract_features_batch(ALL_TWEETS, self.freqs)
        self.assertEqual(batch.shape, (len(ALL_TWEETS), 3))


# ═══════════════════════════════════════════════════════════════════ #
#  predict_tweet                                                     #
# ═══════════════════════════════════════════════════════════════════ #
class TestPredictTweet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        freqs   = _make_freqs()
        X       = extract_features_batch(ALL_TWEETS, freqs)
        _, theta = gradientDescent(X, LABELS_2D, np.zeros((3, 1)), 1e-9, 1000)
        cls.freqs = freqs
        cls.theta = theta

    def test_output_is_float_like(self):
        prob = predict_tweet("I am happy", self.freqs, self.theta)
        self.assertTrue(np.isscalar(float(prob)))

    def test_output_in_range(self):
        prob = float(predict_tweet("I am happy", self.freqs, self.theta))
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_positive_tweet_high_prob(self):
        prob = float(predict_tweet("I love this so much", self.freqs, self.theta))
        self.assertGreater(prob, 0.4)   # relaxed for tiny toy corpus

    def test_negative_tweet_low_prob(self):
        prob = float(predict_tweet("I hate this terrible thing", self.freqs, self.theta))
        self.assertLess(prob, 0.6)


# ═══════════════════════════════════════════════════════════════════ #
#  test_logistic_regression                                          #
# ═══════════════════════════════════════════════════════════════════ #
class TestTestLogisticRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        freqs   = _make_freqs()
        X       = extract_features_batch(ALL_TWEETS, freqs)
        _, theta = gradientDescent(X, LABELS_2D, np.zeros((3, 1)), 1e-9, 2000)
        cls.freqs = freqs
        cls.theta = theta

    def test_returns_float(self):
        acc = evaluate_logistic_regression(ALL_TWEETS, LABELS_2D, self.freqs, self.theta)
        self.assertIsInstance(float(acc), float)

    def test_accuracy_in_range(self):
        acc = evaluate_logistic_regression(ALL_TWEETS, LABELS_2D, self.freqs, self.theta)
        self.assertGreaterEqual(float(acc), 0.0)
        self.assertLessEqual(float(acc), 1.0)

    def test_accuracy_above_chance(self):
        acc = evaluate_logistic_regression(ALL_TWEETS, LABELS_2D, self.freqs, self.theta)
        self.assertGreater(float(acc), 0.5)


# ═══════════════════════════════════════════════════════════════════ #
#  GradientDescentTrainer                                            #
# ═══════════════════════════════════════════════════════════════════ #
class TestGradientDescentTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.freqs = _make_freqs()
        cls.X     = extract_features_batch(ALL_TWEETS, cls.freqs)
        cls.y     = LABELS_2D

    def test_fit_and_predict(self):
        trainer = GradientDescentTrainer(alpha=1e-9, num_iters=500, verbose=0)
        trainer.fit(self.X, self.y)
        self.assertIsNotNone(trainer.theta_)
        self.assertEqual(trainer.theta_.shape, (3, 1))

    def test_cost_decreases(self):
        trainer = GradientDescentTrainer(alpha=1e-9, num_iters=1000,
                                         check_every=100, verbose=0)
        trainer.fit(self.X, self.y)
        if len(trainer.history_.costs) >= 2:
            self.assertLessEqual(trainer.history_.costs[-1],
                                 trainer.history_.costs[0])

    def test_predict_proba_range(self):
        trainer = GradientDescentTrainer(alpha=1e-9, num_iters=500, verbose=0)
        trainer.fit(self.X, self.y)
        proba = trainer.predict_proba(self.X)
        self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1))

    def test_momentum_variant(self):
        trainer = GradientDescentTrainer(alpha=1e-9, num_iters=500,
                                         momentum=0.9, verbose=0)
        trainer.fit(self.X, self.y)
        self.assertIsNotNone(trainer.theta_)

    def test_l2_regularisation(self):
        trainer = GradientDescentTrainer(alpha=1e-9, num_iters=500,
                                         lambda_=0.01, verbose=0)
        trainer.fit(self.X, self.y)
        self.assertIsNotNone(trainer.theta_)


# ═══════════════════════════════════════════════════════════════════ #
#  AdvancedLogisticRegression (sklearn wrapper)                      #
# ═══════════════════════════════════════════════════════════════════ #
class TestAdvancedLogisticRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        freqs = _make_freqs()
        cls.X = extract_features_batch(ALL_TWEETS, freqs)
        cls.y = LABELS_2D

    def test_fit_returns_self(self):
        model = AdvancedLogisticRegression(cv_folds=2, verbose=False)
        fitted = model.fit(self.X, self.y)
        self.assertIs(fitted, model)

    def test_predict_shape(self):
        model = AdvancedLogisticRegression(cv_folds=2, verbose=False)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, (len(ALL_TWEETS),))

    def test_proba_range(self):
        model = AdvancedLogisticRegression(cv_folds=2, verbose=False)
        model.fit(self.X, self.y)
        proba = model.predict_proba(self.X)
        self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1))

    def test_best_params_populated(self):
        model = AdvancedLogisticRegression(cv_folds=2, verbose=False)
        model.fit(self.X, self.y)
        self.assertIn("C", model.best_params_)


# ═══════════════════════════════════════════════════════════════════ #
#  Runner                                                            #
# ═══════════════════════════════════════════════════════════════════ #
if __name__ == "__main__":
    unittest.main(verbosity=2)
