"""
Sample usage script demonstrating how to use the preprocessed tweet data.
Shows various analysis and visualization examples.
"""

import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


def load_preprocessed_data():
    """Load all preprocessed data from disk.
    
    Returns:
        Dictionary containing vectors, tokens, tweets, and features
    """
    print("Loading preprocessed data...")
    
    with open('preprocessed_data/tweet_vectors.pkl', 'rb') as f:
        vectors = pickle.load(f)
    
    with open('preprocessed_data/tweet_tokens.json', 'r') as f:
        tokens = json.load(f)
    
    with open('preprocessed_data/original_tweets.json', 'r') as f:
        tweets = json.load(f)
    
    with open('preprocessed_data/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    with open('preprocessed_data/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded {vectors.shape[0]} tweet vectors")
    print(f"✓ Vocabulary size: {len(feature_names)}")
    print()
    
    return {
        'vectors': vectors,
        'tokens': tokens,
        'tweets': tweets,
        'feature_names': feature_names,
        'metadata': metadata
    }


def analyze_corpus_statistics(data):
    """Print corpus-level statistics.
    
    Args:
        data: Dictionary from load_preprocessed_data()
    """
    print("=" * 80)
    print("CORPUS STATISTICS")
    print("=" * 80)
    
    tokens = data['tokens']
    tweets = data['tweets']
    
    # Token statistics
    token_lengths = [len(t) for t in tokens if t]  # Exclude empty
    vocab_sizes = [len(set(t)) for t in tokens if t]
    
    print(f"Total tweets: {len(tweets)}")
    print(f"Positive tweets: {sum(1 for t in tweets if t['label'] == 'positive')}")
    print(f"Negative tweets: {sum(1 for t in tweets if t['label'] == 'negative')}")
    print()
    print(f"Token statistics (per tweet):")
    print(f"  Min tokens: {min(token_lengths)}")
    print(f"  Max tokens: {max(token_lengths)}")
    print(f"  Avg tokens: {np.mean(token_lengths):.2f}")
    print(f"  Median tokens: {np.median(token_lengths):.2f}")
    print()
    print(f"Vocabulary stats (per tweet):")
    print(f"  Min unique tokens: {min(vocab_sizes)}")
    print(f"  Max unique tokens: {max(vocab_sizes)}")
    print(f"  Avg unique tokens: {np.mean(vocab_sizes):.2f}")
    print()
    
    # Most common words across corpus
    all_tokens = []
    for token_list in tokens:
        all_tokens.extend(token_list)
    
    token_counts = Counter(all_tokens)
    print(f"Top 20 most common tokens:")
    for token, count in token_counts.most_common(20):
        print(f"  {token:20s} : {count:5d} occurrences")
    print()


def find_similar_tweets(data, tweet_idx: int, top_k: int = 5):
    """Find tweets most similar to a given tweet.
    
    Args:
        data: Dictionary from load_preprocessed_data()
        tweet_idx: Index of the query tweet
        top_k: Number of similar tweets to return
    """
    print("=" * 80)
    print(f"FINDING SIMILAR TWEETS TO TWEET #{tweet_idx}")
    print("=" * 80)
    
    vectors = data['vectors']
    tweets = data['tweets']
    
    # Compute similarities
    query_vector = vectors[tweet_idx:tweet_idx+1]
    similarities = cosine_similarity(query_vector, vectors)[0]
    
    # Get top similar indices (excluding the query itself)
    top_indices = np.argsort(similarities)[::-1][:top_k+1][1:]  # Skip self
    
    print(f"\nQuery tweet:\n  {tweets[tweet_idx]['text']}\n")
    print(f"Top {top_k} similar tweets:\n")
    
    for rank, idx in enumerate(top_indices, 1):
        score = similarities[idx]
        label = tweets[idx]['label']
        print(f"{rank}. [Similarity: {score:.4f}, Label: {label}]")
        print(f"   {tweets[idx]['text']}\n")


def analyze_positive_vs_negative(data):
    """Compare positive and negative tweet characteristics.
    
    Args:
        data: Dictionary from load_preprocessed_data()
    """
    print("=" * 80)
    print("POSITIVE vs NEGATIVE TWEETS ANALYSIS")
    print("=" * 80)
    
    tokens = data['tokens']
    tweets = data['tweets']
    
    pos_tokens = []
    neg_tokens = []
    
    for token_list, tweet in zip(tokens, tweets):
        if tweet['label'] == 'positive':
            pos_tokens.extend(token_list)
        else:
            neg_tokens.extend(token_list)
    
    print(f"Positive tweets:")
    print(f"  Total tokens: {len(pos_tokens)}")
    print(f"  Unique tokens: {len(set(pos_tokens))}")
    print(f"  Avg tokens per tweet: {len(pos_tokens) / sum(1 for t in tweets if t['label'] == 'positive'):.2f}")
    print(f"\n  Top 10 tokens:")
    for token, count in Counter(pos_tokens).most_common(10):
        print(f"    {token:15s} : {count:5d}")
    
    print(f"\nNegative tweets:")
    print(f"  Total tokens: {len(neg_tokens)}")
    print(f"  Unique tokens: {len(set(neg_tokens))}")
    print(f"  Avg tokens per tweet: {len(neg_tokens) / sum(1 for t in tweets if t['label'] == 'negative'):.2f}")
    print(f"\n  Top 10 tokens:")
    for token, count in Counter(neg_tokens).most_common(10):
        print(f"    {token:15s} : {count:5d}")
    
    print()


def show_tweet_details(data, tweet_idx: int):
    """Show detailed breakdown of preprocessing for a specific tweet.
    
    Args:
        data: Dictionary from load_preprocessed_data()
        tweet_idx: Index of the tweet
    """
    print("=" * 80)
    print(f"DETAILED PREPROCESSING BREAKDOWN - TWEET #{tweet_idx}")
    print("=" * 80)
    
    tweets = data['tweets']
    tokens = data['tokens']
    
    tweet = tweets[tweet_idx]
    
    print(f"Original Tweet:")
    print(f"  ID: {tweet['id']}")
    print(f"  Label: {tweet['label']}")
    print(f"  Text: {tweet['text']}")
    print(f"  Original word count: {len(tweet['text'].split())}")
    print()
    
    print(f"Preprocessed:")
    print(f"  Tokens: {tokens[tweet_idx]}")
    print(f"  Token count: {len(tokens[tweet_idx])}")
    print(f"  Reduction: {len(tweet['text'].split())} -> {len(tokens[tweet_idx])} words")
    print(f"  Compression: {(1 - len(tokens[tweet_idx])/len(tweet['text'].split())) * 100:.1f}%")
    print()
    
    # Get top TF-IDF features for this tweet
    vectors = data['vectors']
    feature_names = data['feature_names']
    
    vector = vectors[tweet_idx].toarray()[0]
    top_indices = np.argsort(vector)[::-1][:5]
    
    print(f"Top TF-IDF features:")
    for rank, idx in enumerate(top_indices, 1):
        score = vector[idx]
        if score > 0:
            print(f"  {rank}. {feature_names[idx]:20s} : {score:.6f}")
    print()


def main():
    """Run all analysis examples."""
    # Load data
    data = load_preprocessed_data()
    
    # Run analyses
    analyze_corpus_statistics(data)
    analyze_positive_vs_negative(data)
    
    # Show details for a specific tweet
    show_tweet_details(data, tweet_idx=42)
    
    # Find similar tweets
    find_similar_tweets(data, tweet_idx=42, top_k=5)
    
    print("\n✓ All analyses complete!")


if __name__ == "__main__":
    main()
