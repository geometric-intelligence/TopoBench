"""Hostile transport and archive tests for authenticated dataset acquisition."""

from __future__ import annotations

import hashlib
import io
import stat
import tarfile
import threading
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from topobench.data.utils.downloads import (
    CONNECT_TIMEOUT_SECONDS,
    MAX_REDIRECTS,
    READ_TIMEOUT_SECONDS,
    ArchiveLimits,
    RemoteArchive,
    acquire_verified_archive,
)


class _Response:
    def __init__(
        self,
        chunks: Iterable[bytes | BaseException],
        *,
        status_error: requests.HTTPError | None = None,
        content_length: int | None = None,
        status_code: int = 200,
        location: str | None = None,
    ) -> None:
        self._chunks = chunks
        self._status_error = status_error
        self.headers: dict[str, str] = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        if location is not None:
            self.headers["Location"] = location
        self.status_code = status_code
        self.closed = False

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def iter_content(self, chunk_size: int) -> Iterable[bytes]:
        assert chunk_size > 0
        for chunk in self._chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk

    def close(self) -> None:
        self.closed = True


def _zip_bytes(entries: list[tuple[str | zipfile.ZipInfo, bytes]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return stream.getvalue()


def _tar_bytes(member_type: bytes) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo("unsafe")
        member.type = member_type
        if member_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
            member.linkname = "target"
        archive.addfile(member)
    return stream.getvalue()


def _limits(payload: bytes, **overrides: int) -> ArchiveLimits:
    values = {
        "max_compressed_bytes": max(len(payload), 1) + 1024,
        "max_members": 8,
        "max_member_bytes": 1_000_000,
        "max_total_bytes": 2_000_000,
        "max_expansion_ratio": 1_000,
    }
    values.update(overrides)
    return ArchiveLimits(**values)


def _asset(
    payload: bytes,
    *,
    digest: str | None = None,
    size_bytes: int | None = None,
    limits: ArchiveLimits | None = None,
    archive_format: str = "zip",
) -> RemoteArchive:
    return RemoteArchive(
        url="https://assets.example.invalid/hypergraph.zip",
        sha256=digest or hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload) if size_bytes is None else size_bytes,
        limits=limits or _limits(payload),
        archive_format=archive_format,
    )


def _install_response(
    monkeypatch: pytest.MonkeyPatch,
    response: _Response,
) -> None:
    def get(url: str, **kwargs: object) -> _Response:
        assert url.startswith("https://")
        assert kwargs == {
            "stream": True,
            "headers": {"Accept-Encoding": "identity"},
            "timeout": (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            "allow_redirects": False,
        }
        return response

    monkeypatch.setattr(requests, "get", get)


def _validate_fixture(root: Path) -> None:
    assert (root / "dataset" / "data.npz").read_bytes() == b"safe-npz"


def _assert_no_acquisition_leftovers(root: Path) -> None:
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    "field_name",
    [
        "max_compressed_bytes",
        "max_members",
        "max_member_bytes",
        "max_total_bytes",
        "max_expansion_ratio",
    ],
)
def test_archive_limits_require_exact_positive_integer_ceilings(
    field_name: str,
) -> None:
    values = {
        "max_compressed_bytes": 10,
        "max_members": 2,
        "max_member_bytes": 5,
        "max_total_bytes": 10,
        "max_expansion_ratio": 2,
    }
    values[field_name] = 0

    with pytest.raises(ValueError, match=field_name):
        ArchiveLimits(**values)


@pytest.mark.parametrize(
    ("url", "digest", "size_bytes", "message"),
    [
        ("http://example.invalid/archive.zip", "a" * 64, 1, "HTTPS"),
        ("https://example.invalid/archive.zip", "A" * 64, 1, "sha256"),
        ("https://example.invalid/archive.zip", "a" * 64, 0, "size_bytes"),
    ],
)
def test_remote_archive_requires_static_https_digest_and_size(
    url: str,
    digest: str,
    size_bytes: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RemoteArchive(
            url=url,
            sha256=digest,
            size_bytes=size_bytes,
            limits=ArchiveLimits(
                max_compressed_bytes=10,
                max_members=2,
                max_member_bytes=5,
                max_total_bytes=10,
                max_expansion_ratio=2,
            ),
        )


def test_connect_timeout_leaves_no_archive_temp_or_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])

    def timeout(*args: object, **kwargs: object) -> _Response:
        raise requests.ConnectTimeout("connect stalled")

    monkeypatch.setattr(requests, "get", timeout)

    with pytest.raises(requests.ConnectTimeout, match="connect stalled"):
        acquire_verified_archive(_asset(payload), tmp_path / "cache", _validate_fixture)

    _assert_no_acquisition_leftovers(tmp_path)


def test_read_timeout_after_partial_chunk_leaves_no_leftovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    _install_response(
        monkeypatch,
        _Response([payload[:5], requests.ReadTimeout("read stalled")]),
    )

    with pytest.raises(requests.ReadTimeout, match="read stalled"):
        acquire_verified_archive(_asset(payload), tmp_path / "cache", _validate_fixture)

    _assert_no_acquisition_leftovers(tmp_path)


def test_non_success_status_is_checked_before_body_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    response = _Response(
        [AssertionError("body must not be read")],
        status_error=requests.HTTPError("503 Server Error"),
    )
    _install_response(monkeypatch, response)

    with pytest.raises(requests.HTTPError, match="503"):
        acquire_verified_archive(_asset(payload), tmp_path / "cache", _validate_fixture)

    assert response.closed
    _assert_no_acquisition_leftovers(tmp_path)


def test_https_to_http_to_https_chain_never_issues_the_http_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    response = _Response(
        [],
        status_code=302,
        location="http://assets.example.invalid/insecure.zip",
    )
    requested_urls: list[str] = []

    def get(url: str, **kwargs: object) -> _Response:
        requested_urls.append(url)
        if url.startswith("http://"):
            raise AssertionError("HTTP redirect target must never be requested")
        return response

    monkeypatch.setattr(requests, "get", get)

    with pytest.raises(ValueError, match="exact HTTPS URL"):
        acquire_verified_archive(_asset(payload), tmp_path / "cache", _validate_fixture)

    assert requested_urls == ["https://assets.example.invalid/hypergraph.zip"]
    assert response.closed
    _assert_no_acquisition_leftovers(tmp_path)


@pytest.mark.parametrize(
    ("location", "message"),
    [
        (None, "Location"),
        ("https://user@example.invalid/archive.zip", "exact HTTPS URL"),
    ],
)
def test_redirect_requires_a_valid_location_before_another_request(
    location: str | None,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    response = _Response([], status_code=302, location=location)
    requested_urls: list[str] = []

    def get(url: str, **kwargs: object) -> _Response:
        requested_urls.append(url)
        return response

    monkeypatch.setattr(requests, "get", get)

    with pytest.raises(ValueError, match=message):
        acquire_verified_archive(_asset(payload), tmp_path / "cache", _validate_fixture)

    assert len(requested_urls) == 1
    assert response.closed
    _assert_no_acquisition_leftovers(tmp_path)


def test_cyclic_redirect_is_bounded_and_closes_every_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    asset = _asset(payload)
    second_url = "https://assets.example.invalid/second.zip"
    first = _Response([], status_code=302, location=second_url)
    second = _Response([], status_code=302, location=asset.url)
    responses = {asset.url: first, second_url: second}
    requested_urls: list[str] = []

    def get(url: str, **kwargs: object) -> _Response:
        requested_urls.append(url)
        return responses[url]

    monkeypatch.setattr(requests, "get", get)

    with pytest.raises(ValueError, match="redirect cycle"):
        acquire_verified_archive(asset, tmp_path / "cache", _validate_fixture)

    assert requested_urls == [asset.url, second_url]
    assert first.closed and second.closed
    _assert_no_acquisition_leftovers(tmp_path)


def test_redirect_hop_count_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    asset = _asset(payload)
    responses: list[_Response] = []
    requested_urls: list[str] = []

    def get(url: str, **kwargs: object) -> _Response:
        requested_urls.append(url)
        next_hop = len(requested_urls)
        response = _Response(
            [],
            status_code=302,
            location=f"https://assets.example.invalid/hop-{next_hop}.zip",
        )
        responses.append(response)
        return response

    monkeypatch.setattr(requests, "get", get)

    with pytest.raises(ValueError, match="redirect limit"):
        acquire_verified_archive(asset, tmp_path / "cache", _validate_fixture)

    assert len(requested_urls) == MAX_REDIRECTS + 1
    assert all(response.closed for response in responses)
    _assert_no_acquisition_leftovers(tmp_path)


def test_relative_https_redirect_reaches_authenticated_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    asset = _asset(payload)
    redirect = _Response([], status_code=302, location="/final.zip")
    final = _Response([payload])
    responses = iter((redirect, final))
    requested_urls: list[str] = []

    def get(url: str, **kwargs: object) -> _Response:
        requested_urls.append(url)
        return next(responses)

    monkeypatch.setattr(requests, "get", get)

    acquire_verified_archive(asset, tmp_path / "cache", _validate_fixture)

    assert requested_urls == [
        asset.url,
        "https://assets.example.invalid/final.zip",
    ]
    assert redirect.closed and final.closed
    _validate_fixture(tmp_path / "cache")


def test_declared_oversized_body_is_rejected_before_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    limit = len(payload)
    response = _Response(
        [AssertionError("oversized declared body must not be read")],
        content_length=limit + 1,
    )
    _install_response(monkeypatch, response)

    with pytest.raises(ValueError, match="compressed-byte limit"):
        acquire_verified_archive(
            _asset(payload, limits=_limits(payload, max_compressed_bytes=limit)),
            tmp_path / "cache",
            _validate_fixture,
        )

    _assert_no_acquisition_leftovers(tmp_path)


def test_streamed_body_cannot_cross_compressed_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = b"trusted-size"
    response = _Response([trusted, b"unexpected-byte"])
    _install_response(monkeypatch, response)

    with pytest.raises(ValueError, match="compressed-byte limit"):
        acquire_verified_archive(
            _asset(
                trusted,
                limits=_limits(trusted, max_compressed_bytes=len(trusted)),
            ),
            tmp_path / "cache",
            _validate_fixture,
        )

    _assert_no_acquisition_leftovers(tmp_path)


@pytest.mark.parametrize(
    ("digest", "size_bytes", "message"),
    [
        ("0" * 64, None, "SHA-256"),
        (None, 1, "size"),
    ],
)
def test_exact_size_and_digest_are_required_before_extraction(
    digest: str | None,
    size_bytes: int | None,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    _install_response(monkeypatch, _Response([payload]))

    with pytest.raises(ValueError, match=message):
        acquire_verified_archive(
            _asset(payload, digest=digest, size_bytes=size_bytes),
            tmp_path / "cache",
            _validate_fixture,
        )

    _assert_no_acquisition_leftovers(tmp_path)


def test_keyboard_interruption_cleans_private_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    _install_response(monkeypatch, _Response([payload[:5], KeyboardInterrupt()]))

    with pytest.raises(KeyboardInterrupt):
        acquire_verified_archive(_asset(payload), tmp_path / "cache", _validate_fixture)

    _assert_no_acquisition_leftovers(tmp_path)


@pytest.mark.parametrize(
    "unsafe_name",
    [
        ".",
        "./",
        "../escape",
        "/absolute",
        "C:/windows-drive",
        "dir\\windows-separator",
        "dir/../not-normalized",
        "dir//not-normalized",
        "dir/./not-normalized",
    ],
)
def test_zip_member_paths_must_be_normalized_and_contained(
    unsafe_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([(unsafe_name, b"unsafe")])
    _install_response(monkeypatch, _Response([payload]))

    with pytest.raises(ValueError, match="archive member path"):
        acquire_verified_archive(_asset(payload), tmp_path / "cache", lambda _: None)

    _assert_no_acquisition_leftovers(tmp_path)


@pytest.mark.parametrize(
    "member_type",
    [
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.FIFOTYPE,
    ],
)
def test_tar_rejects_links_devices_and_fifos_before_extraction(
    member_type: bytes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _tar_bytes(member_type)
    _install_response(monkeypatch, _Response([payload]))

    with pytest.raises(ValueError, match="regular file or directory"):
        acquire_verified_archive(
            _asset(payload, archive_format="tar"),
            tmp_path / "cache",
            lambda _: None,
        )

    _assert_no_acquisition_leftovers(tmp_path)


def test_zip_rejects_non_regular_unix_member_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member = zipfile.ZipInfo("link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    payload = _zip_bytes([(member, b"target")])
    _install_response(monkeypatch, _Response([payload]))

    with pytest.raises(ValueError, match="regular file or directory"):
        acquire_verified_archive(_asset(payload), tmp_path / "cache", lambda _: None)

    _assert_no_acquisition_leftovers(tmp_path)


def test_archive_member_count_is_bounded_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("one", b"1"), ("two", b"2")])
    _install_response(monkeypatch, _Response([payload]))

    with pytest.raises(ValueError, match="member count"):
        acquire_verified_archive(
            _asset(payload, limits=_limits(payload, max_members=1)),
            tmp_path / "cache",
            lambda _: None,
        )

    _assert_no_acquisition_leftovers(tmp_path)


def test_each_expanded_member_is_bounded_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("large", b"x" * 10)])
    _install_response(monkeypatch, _Response([payload]))

    with pytest.raises(ValueError, match="per-member"):
        acquire_verified_archive(
            _asset(payload, limits=_limits(payload, max_member_bytes=9)),
            tmp_path / "cache",
            lambda _: None,
        )

    _assert_no_acquisition_leftovers(tmp_path)


def test_total_expanded_bytes_use_a_bounded_accumulator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("one", b"x" * 6), ("two", b"y" * 6)])
    _install_response(monkeypatch, _Response([payload]))

    with pytest.raises(ValueError, match="total expanded"):
        acquire_verified_archive(
            _asset(
                payload,
                limits=_limits(
                    payload,
                    max_member_bytes=10,
                    max_total_bytes=10,
                ),
            ),
            tmp_path / "cache",
            lambda _: None,
        )

    _assert_no_acquisition_leftovers(tmp_path)


def test_total_expansion_ratio_is_bounded_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("bomb", b"0" * 100_000)])
    _install_response(monkeypatch, _Response([payload]))

    with pytest.raises(ValueError, match="expansion ratio"):
        acquire_verified_archive(
            _asset(payload, limits=_limits(payload, max_expansion_ratio=2)),
            tmp_path / "cache",
            lambda _: None,
        )

    _assert_no_acquisition_leftovers(tmp_path)


def test_safe_format_callback_failure_prevents_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"not-safe")])
    _install_response(monkeypatch, _Response([payload]))

    def reject_format(root: Path) -> None:
        assert root != tmp_path / "cache"
        raise ValueError("unsafe format schema")

    with pytest.raises(ValueError, match="unsafe format schema"):
        acquire_verified_archive(_asset(payload), tmp_path / "cache", reject_format)

    _assert_no_acquisition_leftovers(tmp_path)


def test_valid_directory_entry_after_its_child_extracts_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes(
        [
            ("dataset/data.npz", b"safe-npz"),
            ("dataset/", b""),
        ]
    )
    destination = tmp_path / "cache"
    _install_response(monkeypatch, _Response([payload]))

    acquire_verified_archive(_asset(payload), destination, _validate_fixture)

    _validate_fixture(destination)
    assert {path.name for path in tmp_path.iterdir()} == {"cache"}


def test_valid_fixture_is_visible_only_after_atomic_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    destination = tmp_path / "cache"
    _install_response(monkeypatch, _Response([payload]))
    validated_root: Path | None = None

    def validate(root: Path) -> None:
        nonlocal validated_root
        validated_root = root
        assert root != destination
        assert not destination.exists()
        _validate_fixture(root)

    result = acquire_verified_archive(_asset(payload), destination, validate)

    assert result == destination
    assert validated_root is not None
    assert not validated_root.exists()
    _validate_fixture(destination)
    assert {path.name for path in tmp_path.iterdir()} == {"cache"}


def test_concurrent_attempts_publish_one_complete_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    destination = tmp_path / "cache"
    request_count = 0
    count_lock = threading.Lock()

    def get(url: str, **kwargs: object) -> _Response:
        nonlocal request_count
        with count_lock:
            request_count += 1
        return _Response([payload])

    monkeypatch.setattr(requests, "get", get)
    asset = _asset(payload)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda _: acquire_verified_archive(
                    asset,
                    destination,
                    _validate_fixture,
                ),
                range(2),
            )
        )

    assert results == (destination, destination)
    assert request_count == 1
    _validate_fixture(destination)
    assert {path.name for path in tmp_path.iterdir()} == {"cache"}


def test_unauthenticated_partial_cache_is_replaced_without_mixing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    destination = tmp_path / "cache"
    destination.mkdir()
    (destination / "partial").write_bytes(b"untrusted-prefix")
    _install_response(monkeypatch, _Response([payload]))

    acquire_verified_archive(_asset(payload), destination, _validate_fixture)

    assert not (destination / "partial").exists()
    _validate_fixture(destination)
    assert {path.name for path in tmp_path.iterdir()} == {"cache"}


def test_interrupted_attempt_is_never_resumed_or_mixed_with_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _zip_bytes([("dataset/data.npz", b"safe-npz")])
    destination = tmp_path / "cache"
    responses = iter(
        (
            _Response([payload[:10], requests.ReadTimeout("partial")]),
            _Response([payload]),
        )
    )

    def get(url: str, **kwargs: object) -> _Response:
        return next(responses)

    monkeypatch.setattr(requests, "get", get)
    asset = _asset(payload)

    with pytest.raises(requests.ReadTimeout, match="partial"):
        acquire_verified_archive(asset, destination, _validate_fixture)
    assert not destination.exists()

    assert (
        acquire_verified_archive(asset, destination, _validate_fixture)
        == destination
    )
    _validate_fixture(destination)
    assert {path.name for path in tmp_path.iterdir()} == {"cache"}
