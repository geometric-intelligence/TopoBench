# Changelog

## Unreleased

### Security

- Replaced executable PyG processed-cache object loading with versioned tensor/primitive payloads, static `Data`/`HeteroData`/`HypergraphData` reconstruction, digest manifests, atomic publication, and per-principal trusted-root checks.
- Split checkpoint loading policy: selected-model evaluation uses digest-bound `weights_only=True` state dictionaries with strict key matching; qualified full resume accepts only private, same-run checkpoints validated against their recorded path, size, and SHA-256 before Lightning deserialization.
- Rejected arbitrary external or stale checkpoint paths before pipeline, model, or trainer construction in qualified execution.

### Build

- Made environment setup immutable with `uv sync --frozen --all-extras`; setup no longer edits `pyproject.toml`, deletes `uv.lock`, or installs unreviewed packages after synchronization.
- Pinned every GitHub Action to a reviewed commit SHA, disabled persisted checkout credentials, and moved permissions to least-privilege job scopes.
