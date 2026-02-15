# Tweet Preprocessing & Vectorization Pipeline

A production-grade, architecturally robust pipeline for preprocessing and vectorizing tweets from the NLTK corpus with detailed logging and comprehensive output storage.

## Pipeline Architecture

The pipeline is organized into modular, composable components:

### 1. **config.py** - Configuration Management
Centralized configuration for all preprocessing and vectorization parameters:
- Preprocessing settings (case preservation, handle stripping, length reduction)
- Stopwords language configuration
- TF-IDF vectorizer parameters
- Output file paths
- Logging settings

### 2. **tweet_preprocessor.py** - Core Preprocessing Engine
`TweetPreprocessor` class with 7-step preprocessing pipeline:
1. **Remove RT Markers** - Strip retweet prefixes (`RT @user: ...`)
2. **Remove Hyperlinks** - Remove HTTP(S) URLs
3. **Remove Hash Symbols** - Strip `#` but preserve hashtag text
4. **Remove Extra Whitespace** - Normalize whitespace
5. **Tokenization** - Use NLTK TweetTokenizer (preserves tweet-specific syntax)
6. **Remove Stopwords & Punctuation** - Filter common words and punctuation
7. **Stemming** - Apply Porter Stemmer to normalize word forms

Each step produces detailed metadata about the transformation for auditability.

### 3. **vectorizer.py** - TF-IDF Vectorization Engine
`TweetVectorizer` class built on scikit-learn with:
- TF-IDF sparse matrix representation
- Configurable N-gram support (1-2 grams by default)
- Feature extraction with learned vocabulary
- Model persistence (save/load)
- Top feature extraction per document

### 4. **data_processor.py** - Orchestration Pipeline
`TweetCorpusProcessor` class that:
- Loads entire NLTK tweet corpus (10,000 tweets: 5,000 positive, 5,000 negative)
- Applies preprocessing to all tweets with progress tracking
- Vectorizes all tweets using learned TF-IDF model
- Persists all results to disk
- Provides comprehensive logging and metadata

## Processing Results

### Generated Files

After running the pipeline, the `preprocessed_data/` directory contains:

| File | Description | Size |
|------|-------------|------|
| `tweet_tokens.json` | List of preprocessed token lists (one per tweet) | ~894 KB |
| `tweet_vectors.pkl` | Sparse TF-IDF matrix (10,000 × 5,000) | ~702 KB |
| `original_tweets.json` | Original tweets with labels (positive/negative) | ~1.3 MB |
| `feature_names.json` | List of 5,000 vocabulary terms | ~63 KB |
| `vectorizer.pkl` | Trained TF-IDF vectorizer (for new tweets) | ~181 KB |
| `metadata.json` | Processing and vectorization statistics | ~1 KB |
| `processing_log.txt` | Detailed pipeline execution log | ~3 KB |

### Processing Statistics

```
Total Tweets: 10,000 (5,000 positive, 5,000 negative)
Processing Success Rate: 100% (0 failed)
Average Tokens Before Processing: 11.64
Average Tokens After Processing: 6.74
Vocabulary Size: 5,000 terms
Vector Shape: (10,000 tweets × 5,000 features)
Matrix Sparsity: 99.89% (efficient storage)
```

## Usage

### Basic Run

```bash
# Activate virtual environment (optional)
source .venv/bin/activate

# Run the complete pipeline
python data_processor.py
```

### Advanced Usage

Create a custom script to use the preprocessed data:

```python
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed data
with open('preprocessed_data/tweet_vectors.pkl', 'rb') as f:
    vectors = pickle.load(f)

with open('preprocessed_data/original_tweets.json', 'r') as f:
    tweets = json.load(f)

with open('preprocessed_data/tweet_tokens.json', 'r') as f:
    tokens = json.load(f)

# Example: Find similar tweets
idx = 0
similarities = cosine_similarity(vectors[idx:idx+1], vectors)[0]
top_similar_indices = similarities.argsort()[-5:][::-1]

print(f"Tweet: {tweets[idx]['text']}")
print("\nMost similar tweets:")
for i in top_similar_indices[1:]:  # Skip the first (itself)
    print(f"  - {tweets[i]['text']}")
```

### Load and Use Individual Components

```python
from tweet_preprocessor import TweetPreprocessor
from vectorizer import TweetVectorizer
import pickle

# Load the trained vectorizer
with open('preprocessed_data/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Process a new tweet
preprocessor = TweetPreprocessor()
raw_tweet = "Just deployed my new ML model! #DeepLearning https://example.com"
tokens = preprocessor.process(raw_tweet)
print(tokens)  # ['deploy', 'new', 'ml', 'model', 'deeplearn']

# Vectorize the new tweet
document = " ".join(tokens)
vector = vectorizer.transform([document])
print(vector.shape)  # (1, 5000)
```

## Pipeline Features

### Robustness

✅ **Error Handling**: Gracefully handles malformed tweets  
✅ **Comprehensive Logging**: All steps logged to file and console  
✅ **Progress Tracking**: tqdm progress bars for long operations  
✅ **Result Verification**: Metadata validation for all outputs  
✅ **Reproducibility**: All configuration centralized in `config.py`

### Efficiency

✅ **Sparse Vectorization**: Uses sparse matrices for memory efficiency  
✅ **Batch Processing**: Processes all 10,000 tweets efficiently  
✅ **Modular Design**: Components can be used independently  
✅ **Fast Tokenization**: NLTK TweetTokenizer optimized for Twitter text

### Architectural Quality

✅ **Separation of Concerns**: Each module has single responsibility  
✅ **Configurability**: Centralized config for easy parameter tuning  
✅ **Extensibility**: Easy to add new preprocessing steps or vectorization methods  
✅ **Type Hints**: Full type annotations for IDE support and clarity  
✅ **Comprehensive Documentation**: Detailed docstrings for all classes and methods

## Dependencies

```
nltk              # NLP toolkit
matplotlib        # Visualization (optional)
scikit-learn      # Machine learning
scipy             # Scientific computing
tqdm              # Progress bars
```

Install via:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to customize:

- **Preprocessing Parameters**: Token case preservation, handle stripping, etc.
- **TF-IDF Settings**: Max features, min/max document frequency, N-gram range
- **Output Paths**: Change where results are saved
- **Logging**: Adjust log level and format

## Output Structure

```
preprocessed_data/
├── tweet_tokens.json          # All preprocessed tokens
├── tweet_vectors.pkl          # Sparse TF-IDF matrix
├── original_tweets.json       # Original texts with labels
├── feature_names.json         # Vocabulary (5,000 terms)
├── vectorizer.pkl            # Trained TF-IDF model
├── metadata.json             # Processing statistics
└── processing_log.txt        # Execution log
```

## Performance

- **Processing Time**: ~1.6 seconds for 10,000 tweets
- **Throughput**: ~6,250 tweets/second
- **Vectorization Time**: ~100ms for 10,000 tweets
- **Compression Ratio**: 99.89% sparsity (sparse storage)

## Extension Examples

### Add Custom Preprocessing Step

```python
# In tweet_preprocessor.py, add method:
def _remove_emojis(self, tweet: str) -> Tuple[str, dict]:
    """Remove emoji characters"""
    processed = re.sub(r'[^\w\s#@]', '', tweet)
    return processed, {"step": "remove_emojis"}
```

### Use Different Vectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# In vectorizer.py, replace TfidfVectorizer with:
self.vectorizer = CountVectorizer(...)  # Bag-of-words instead
```

### Add Classification Model

```python
from sklearn.linear_model import LogisticRegression

# Train on vectors
from data_processor import TweetCorpusProcessor
processor = TweetCorpusProcessor()
results = processor.run_pipeline()

vectors = results['vectors']
labels = [1 if item['label'] == 'positive' else 0 
          for item in results['tweets']]

model = LogisticRegression()
model.fit(vectors, labels)
```

## Troubleshooting

**Q: NLTK data missing?**  
A: The pipeline auto-downloads required corpora on first run.

**Q: Memory issues with large corpus?**  
A: Sparse matrices are already used. For even larger corpora, consider batch processing.

**Q: Want to reprocess with different settings?**  
A: Edit `config.py` and run `python data_processor.py` again.

---

**Created**: February 2026  
**Architecture**: Modular, Scalable, Production-Ready  
**Test Coverage**: Full pipeline tested with 10,000 tweets
