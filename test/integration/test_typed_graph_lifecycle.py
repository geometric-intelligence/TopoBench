"""Fresh, cached, downloaded, moved, quarantined, and clean-process lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from test.data.stores.test_topology_only_pyg_partitioner import (
    asymmetric_typed_source,
)
from test.data.stores.test_typed_graph_store import _build_qualified_store
from topobench.data.stores.store_bundle import StoreBundle
from topobench.data.stores.typed_graph_store import (
    TypedGraphStore,
    TypedGraphStoreWriter,
)


PROJECT_ROOT = Path(__file__).parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _writable(path: Path) -> None:
    for item in (path, *path.rglob("*")):
        if not item.is_symlink():
            item.chmod(0o755 if item.is_dir() else 0o644)


def test_store_lifecycle_preserves_one_content_identity_across_every_entry_path(
    tmp_path: Path,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    fixture = _build_qualified_store(source, tmp_path / "stores")
    identity = fixture.store_build.content_sha256
    fresh_path = fixture.store_build.path
    assert fixture.store_build.cache_hit is False

    cached = TypedGraphStoreWriter(
        fixture.ingestor,
        fixture.partition_build,
    ).build()
    assert cached.cache_hit is True
    assert cached.content_sha256 == identity
    assert cached.path == fresh_path
    cached.store.close()
    fixture.store_build.store.close()

    _writable(fresh_path)
    payload = fresh_path / "nodes/n0000/x.npy"
    payload.write_bytes(payload.read_bytes() + b"corrupt")
    rebuilt = TypedGraphStoreWriter(
        fixture.ingestor,
        fixture.partition_build,
    ).build()
    quarantines = tuple(
        fixture.ingestor.store_root.glob(f".quarantine-{identity}-*")
    )
    assert rebuilt.cache_hit is False
    assert rebuilt.content_sha256 == identity
    assert len(quarantines) == 1
    assert quarantines[0].is_dir()
    rebuilt.store.close()

    archive = tmp_path / "prepartitioned.zip"
    artifact = StoreBundle.package(rebuilt.path, archive)
    assert artifact.sha256 == _sha256(archive)
    downloaded = StoreBundle.download(
        archive.as_uri(),
        tmp_path / "downloaded.zip",
        expected_sha256=artifact.sha256,
        max_bytes=artifact.byte_size,
    )
    installed = StoreBundle.install(
        downloaded,
        tmp_path / "installed",
        expected_sha256=artifact.sha256,
    )
    assert installed.content_sha256 == identity
    installed_path = installed.path
    installed.close()

    reopened = TypedGraphStore.open(installed_path)
    moved = StoreBundle.move(reopened, tmp_path / "moved")
    moved_path = moved.path
    assert moved.content_sha256 == identity
    assert moved_path.name == identity
    assert not installed_path.exists()
    assert moved.external_ids("author", [0, 3]) == ["a", "d"]
    moved.close()

    script = """
import json
import sys
from pathlib import Path
from topobench.data.stores.typed_graph_store import TypedGraphStore

path = Path(sys.argv[1])
expected = sys.argv[2]
with TypedGraphStore.open(path) as store:
    assert store.content_sha256 == expected
    assert store.partition_book_identity
    assert store.active_split_tag == 'primary'
    result = {
        'content_sha256': store.content_sha256,
        'partition_book_identity': store.partition_book_identity,
        'active_split_tag': store.active_split_tag,
        'external_ids': store.external_ids('author', [0, 3]),
        'mapped_paths': list(store.mapped_paths),
    }
print(json.dumps(result, sort_keys=True))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(PROJECT_ROOT), environment.get("PYTHONPATH", ""))
        if value
    )
    clean = subprocess.run(
        [sys.executable, "-c", script, str(moved_path), identity],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert clean.returncode == 0, clean.stderr
    observed = json.loads(clean.stdout)
    assert observed == {
        "active_split_tag": "primary",
        "content_sha256": identity,
        "external_ids": ["a", "d"],
        "mapped_paths": [],
        "partition_book_identity": fixture.partition_build.book.content_identity,
    }

    with TypedGraphStore.open(moved_path) as final:
        assert final.content_sha256 == identity
        assert final.partition_book_identity == observed["partition_book_identity"]
