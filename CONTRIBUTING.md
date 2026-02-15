# Contributing Guide

Thank you for your interest in contributing to the Sentiment Analysis Pipeline project!

## Getting Started

### Prerequisites
- Python 3.9+
- Git
- Virtual environment (recommended)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ervardaan/Sentiment-analysis.git
cd Sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pre-commit pytest pytest-cov

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

Use branch naming convention:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `chore/` - Maintenance tasks
- `perf/` - Performance improvements

### 2. Make Changes

Before committing, run:
```bash
# Format code
black .
isort .

# Lint
flake8 .
pylint tweet_preprocessing.py

# Type check
mypy .

# Run tests
pytest tests/ -v --cov

# Pre-commit hook
pre-commit run --all-files
```

### 3. Commit Message Convention

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

Example:
```
feat(preprocessing): add emoji removal

Added support for removing emoji characters from tweets
during preprocessing phase.

Closes #42
```

### 4. Push and Create Pull Request

```bash
git push -u origin feature/your-feature-name
```

Then create a PR on GitHub:
- Describe the changes
- Reference related issues
- Add screenshots/benchmarks if applicable
- Ensure CI passes

### 5. Code Review & Merge

- At least 1 approval required
- All CI checks must pass
- Squash and merge to keep history clean

## Testing

### Run Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=. --cov-report=html

# Specific test
pytest tests/test_preprocessing.py::TestTweetPreprocessor -v
```

### Write Tests
- Place test files in `tests/` directory
- Use naming convention: `test_*.py`
- Use descriptive test names
- Aim for >80% code coverage

## Code Quality Standards

- **PEP 8**: Follow Python style guide
- **Type Hints**: Use type annotations
- **Docstrings**: Document all public functions
- **Comments**: Explain complex logic
- **Max line length**: 120 characters

## Documentation

Update documentation when:
- Adding new features
- Changing API
- Fixing bugs with user impact

Documentation location: `docs/` and `README.md`

## Performance

When adding features:
- Run benchmarks: `python -m pytest benchmarks/`
- Compare with main branch
- Document performance impact
- Seek guidance if significant regression

## Security

- Do not commit secrets (API keys, tokens)
- Use `.env` file for local config
- Report security issues privately to maintainers
- Update dependencies regularly

## Questions?

- Check existing issues and discussions
- Create a new GitHub issue
- Join our community discussions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT).

Thank you for contributing! 🎉
