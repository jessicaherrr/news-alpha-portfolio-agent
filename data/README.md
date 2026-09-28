# Data directory

Raw and processed market data are intentionally git-ignored.

- `raw/`: immutable vendor downloads
- `processed/`: canonical normalized datasets
- `cache/`: disposable caches

Commit only manifests/metadata that do not contain secrets or licensed raw data.
