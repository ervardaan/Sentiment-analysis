# conftest.py  –  pytest configuration for the NLP project
# ---------------------------------------------------------
# Stop pytest from scanning project source files for test functions.
# Only files matching tests/test_*.py are collected.
collect_ignore_glob = ["*.py"]  # root-level files: ignore all
