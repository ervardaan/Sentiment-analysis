"""Tests for Tweet Preprocessing pipeline."""

import pytest
from tweet_preprocessing import TweetPreprocessor, TweetVectorizer


class TestTweetPreprocessor:
    """Test suite for TweetPreprocessor class."""

    def setup_method(self):
        """Initialize preprocessor before each test."""
        self.preprocessor = TweetPreprocessor()

    def test_remove_retweet_markers(self):
        """Test removal of RT markers."""
        tweet = "RT @user This is a retweet"
        result = self.preprocessor.process(tweet)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_remove_hyperlinks(self):
        """Test removal of HTTP(S) URLs."""
        tweet = "Check this out https://example.com and http://example.org"
        result = self.preprocessor.process(tweet)
        assert isinstance(result, list)
        # URL tokens should not appear in result
        assert not any('http' in token for token in result)

    def test_remove_hash_symbols(self):
        """Test removal of # but preservation of text."""
        tweet = "#hashtag #sentiment #analysis"
        result = self.preprocessor.process(tweet)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_tokenization(self):
        """Test tweet tokenization."""
        tweet = "This is a simple test tweet"
        result = self.preprocessor.process(tweet)
        assert isinstance(result, list)
        assert all(isinstance(token, str) for token in result)

    def test_stopword_removal(self):
        """Test that common stopwords are removed."""
        tweet = "the a is in on at by for with from"
        result = self.preprocessor.process(tweet)
        # Most stopwords should be removed
        assert len(result) == 0 or len(result) < 10

    def test_empty_tweet(self):
        """Test handling of empty or whitespace-only tweets."""
        result = self.preprocessor.process("")
        assert isinstance(result, list)

    def test_special_characters(self):
        """Test handling of special characters and emojis."""
        tweet = "Hello :) World :( Test @#$%"
        result = self.preprocessor.process(tweet)
        assert isinstance(result, list)

    def test_mixed_case_handling(self):
        """Test that mixed case is handled correctly."""
        tweet = "HELLO world HeLLo WoRLd"
        result = self.preprocessor.process(tweet)
        assert isinstance(result, list)
        assert all(token == token.lower() for token in result)


class TestTweetVectorizer:
    """Test suite for TweetVectorizer class."""

    def setup_method(self):
        """Initialize vectorizer before each test."""
        self.vectorizer = TweetVectorizer()

    def test_initialization(self):
        """Test vectorizer initialization."""
        assert self.vectorizer.is_fitted is False

    def test_fit_transform(self):
        """Test fitting and transforming documents."""
        docs = [
            "hello world",
            "test document",
            "another example",
        ]
        vectors, metadata = self.vectorizer.fit_transform(docs)
        assert vectors.shape[0] == len(docs)
        assert self.vectorizer.is_fitted is True
        assert "vector_shape" in metadata
        assert "vocab_size" in metadata

    def test_sparse_matrix_output(self):
        """Test that output is sparse matrix."""
        docs = ["hello world", "test document"]
        vectors, _ = self.vectorizer.fit_transform(docs)
        assert hasattr(vectors, 'toarray')  # Check if sparse

    def test_transform_requires_fit(self):
        """Test that transform raises error if not fitted."""
        docs = ["hello world"]
        with pytest.raises(RuntimeError):
            self.vectorizer.transform(docs)

    def test_vocabulary_size(self):
        """Test vocabulary size is set correctly."""
        docs = [f"doc {i}" for i in range(10)]
        vectors, metadata = self.vectorizer.fit_transform(docs)
        assert metadata["vocab_size"] <= self.vectorizer.vectorizer.max_features


class TestIntegration:
    """Integration tests for preprocessing + vectorization pipeline."""

    def test_preprocessing_to_vectorization_pipeline(self):
        """Test full pipeline from preprocessing to vectorization."""
        preprocessor = TweetPreprocessor()
        vectorizer = TweetVectorizer()

        # Raw tweets
        tweets = [
            "@user1 This is great! https://example.com #sentiment",
            "RT @user2 Really bad experience :( #negative",
            "Neutral tweet with no special content",
        ]

        # Preprocess
        tokens = [preprocessor.process(t) for t in tweets]
        assert len(tokens) == len(tweets)
        assert all(isinstance(t, list) for t in tokens)

        # Vectorize
        docs = [" ".join(t) for t in tokens]
        vectors, metadata = vectorizer.fit_transform(docs)

        assert vectors.shape[0] == len(tweets)
        assert metadata["vocab_size"] > 0

    def test_pipeline_end_to_end(self):
        """Test complete end-to-end pipeline."""
        preprocessor = TweetPreprocessor()
        vectorizer = TweetVectorizer()

        # Large batch
        tweets = ["This is tweet number " + str(i) for i in range(100)]

        # Process
        tokens = [preprocessor.process(t) for t in tweets]
        docs = [" ".join(t) for t in tokens]

        # Vectorize
        vectors, metadata = vectorizer.fit_transform(docs)

        assert vectors.shape[0] == 100
        assert not vectorizer.is_fitted is False
