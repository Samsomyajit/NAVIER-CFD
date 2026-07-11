# Contributing

1. Add or update the executable model/dataset catalog.
2. Add a Markdown architecture, dataset, or case-study card under `docs/`.
3. Add or update tests.
4. Never vendor external model code without confirming its license.
5. Report upstream commit hashes and dataset revisions.
6. Distinguish metadata registration from an executable, tested adapter.

```bash
pip install -e ".[dev]"
ruff check src tests
pytest
```
