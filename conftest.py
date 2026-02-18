# conftest.py  –  pytest configuration for the NLP project
# ---------------------------------------------------------
# Stop pytest from scanning large data or generated files for tests.
# Restrict ignored patterns to non-source directories only.
collect_ignore_glob = ["preprocessed_data/*"]
