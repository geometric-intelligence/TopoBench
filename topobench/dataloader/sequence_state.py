"""Commit-safe ordered sequence state for deterministic graph sampling."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import PurePath
from typing import Any, TypeAlias

import torch
from torch import Tensor

CanonicalScalar: TypeAlias = None | bool | int | float | str
CanonicalValue: TypeAlias = CanonicalScalar | tuple["CanonicalValue", ...]
CanonicalItems: TypeAlias = tuple[tuple[str, CanonicalValue], ...]


_IDENTITY_FORMAT = "sampler-sequence-identity-v1"
_STATE_FORMAT = "sampler-sequence-state-v1"
_SHA256_LENGTH = 64
_FORBIDDEN_METADATA_KEYS = frozenset(
    {"host", "hostname", "pid", "process", "process_id", "worker", "worker_id"}
)
_IDENTITY_KEYS = frozenset(
    {
        "format_version", "store_content_sha256", "partition_book_identity",
        "active_split_tag", "fitted_transform_state_key", "sampling_strategy_type",
        "sampling_strategy_config", "sampling_strategy_fingerprint",
        "phase_descriptor_digest", "phase_descriptor_order", "sampler_rng_identity",
    }
)
_STATE_KEYS = frozenset(
    {
        "format_version", "identity", "committed_cursor", "committed_rng_state",
        "committed_sampler_state", "committed_evaluator_sequence",
        "committed_evaluator_count", "committed_evaluator_state",
        "committed_global_step", "committed_epoch",
    }
)


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise ValueError(f"{name} must be a SHA-256 identity")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 identity") from error
    return value.lower()


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _canonical_value(value: object, name: str, *, metadata_key: str | None = None) -> CanonicalValue:
    if metadata_key is not None and metadata_key.lower() in _FORBIDDEN_METADATA_KEYS:
        raise ValueError(f"{name} contains forbidden process metadata {metadata_key!r}")
    if isinstance(value, str):
        if PurePath(value).is_absolute():
            raise ValueError(f"{name} cannot contain absolute paths")
        return value
    if value is None or type(value) is bool:
        return value
    if isinstance(value, PurePath):
        raise TypeError(f"{name} cannot contain filesystem paths")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} cannot contain non-finite numbers")
        return result
    if isinstance(value, Mapping):
        items: list[tuple[str, CanonicalValue]] = []
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{name} mapping keys must be non-empty strings")
            items.append((key, _canonical_value(item, name, metadata_key=key)))
        return tuple(sorted(items))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_canonical_value(item, name) for item in value)
    raise TypeError(f"{name} must contain only canonical scalar, tuple, or mapping values")


def _canonical_items(value: object, name: str) -> CanonicalItems:
    if isinstance(value, Mapping):
        source = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        source = tuple(value)
    else:
        raise TypeError(f"{name} must be a mapping or tuple of pairs")
    items: list[tuple[str, CanonicalValue]] = []
    for item in source:
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes, bytearray))
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
        ):
            raise TypeError(f"{name} must contain string-keyed pairs")
        items.append(
            (
                item[0],
                _canonical_value(item[1], name, metadata_key=item[0]),
            )
        )
    if len({key for key, _ in items}) != len(items):
        raise ValueError(f"{name} contains duplicate keys")
    return tuple(sorted(items))


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _descriptor_record(descriptor: object) -> dict[str, object]:
    names = (
        "content_sha256", "active_split_tag", "phase", "strategy",
        "strategy_options_json", "batch_ordinal", "partition_ids",
        "target_node_type", "target_seed_ids", "participant_counts",
        "generator_seed", "generator_state_sha256",
    )
    missing = tuple(name for name in names if not hasattr(descriptor, name))
    if missing:
        raise TypeError(f"sampling descriptor is missing fields {missing!r}")
    return {name: getattr(descriptor, name) for name in names}


def _descriptor_fingerprint(descriptor: object) -> str:
    return _digest(_descriptor_record(descriptor))


def _clone_checkpoint_value(value: object, name: str) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} cannot contain non-finite numbers")
        return value
    if isinstance(value, Tensor):
        if value.layout != torch.strided:
            raise TypeError(f"{name} tensors must use strided layout")
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{name} mapping keys must be non-empty strings")
            output[key] = _clone_checkpoint_value(item, name)
        return output
    if isinstance(value, tuple):
        return tuple(_clone_checkpoint_value(item, name) for item in value)
    if isinstance(value, list):
        return [_clone_checkpoint_value(item, name) for item in value]
    raise TypeError(f"{name} must contain only JSON scalars, sequences, mappings, or tensors")


def _exact_keys(value: object, expected: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys must match exactly; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    return value


@dataclass(frozen=True, slots=True)
class SequenceIdentity:
    """Immutable content and schedule identity for one training sequence."""

    store_content_sha256: str
    partition_book_identity: str
    active_split_tag: str
    fitted_transform_state_key: str | None
    sampling_strategy_type: str
    sampling_strategy_config: CanonicalItems
    sampling_strategy_fingerprint: str
    phase_descriptor_digest: str
    phase_descriptor_order: tuple[str, ...]
    sampler_rng_identity: CanonicalItems
    format_version: str = _IDENTITY_FORMAT

    def __post_init__(self) -> None:
        if self.format_version != _IDENTITY_FORMAT:
            raise ValueError(f"sequence identity version must be {_IDENTITY_FORMAT!r}")
        object.__setattr__(self, "store_content_sha256", _sha256(self.store_content_sha256, "store_content_sha256"))
        object.__setattr__(self, "partition_book_identity", _sha256(self.partition_book_identity, "partition_book_identity"))
        _nonempty(self.active_split_tag, "active_split_tag")
        if self.fitted_transform_state_key is not None:
            object.__setattr__(self, "fitted_transform_state_key", _sha256(self.fitted_transform_state_key, "fitted_transform_state_key"))
        _nonempty(self.sampling_strategy_type, "sampling_strategy_type")
        config = _canonical_items(self.sampling_strategy_config, "sampling_strategy_config")
        sampler = _canonical_items(self.sampler_rng_identity, "sampler_rng_identity")
        object.__setattr__(self, "sampling_strategy_config", config)
        object.__setattr__(self, "sampler_rng_identity", sampler)
        order = tuple(_sha256(value, "phase descriptor fingerprint") for value in self.phase_descriptor_order)
        if not order:
            raise ValueError("phase_descriptor_order cannot be empty")
        object.__setattr__(self, "phase_descriptor_order", order)
        strategy_fingerprint = _digest({"config": config, "type": self.sampling_strategy_type})
        if self.sampling_strategy_fingerprint != strategy_fingerprint:
            raise ValueError("sampling strategy fingerprint mismatch")
        if self.phase_descriptor_digest != _digest(order):
            raise ValueError("phase descriptor digest mismatch")
        _sha256(self.sampling_strategy_fingerprint, "sampling_strategy_fingerprint")
        _sha256(self.phase_descriptor_digest, "phase_descriptor_digest")

    @classmethod
    def from_descriptors(
        cls,
        descriptors: Sequence[object],
        *,
        partition_book_identity: str,
        fitted_transform_state_key: str | None,
        sampler_state: Mapping[str, object],
    ) -> "SequenceIdentity":
        values = tuple(descriptors)
        if not values:
            raise ValueError("training descriptor schedule cannot be empty")
        records = tuple(_descriptor_record(descriptor) for descriptor in values)
        first = records[0]
        if first["phase"] != "train":
            raise ValueError("sequence identity requires train phase descriptors")
        try:
            strategy_config = json.loads(str(first["strategy_options_json"]))
        except json.JSONDecodeError as error:
            raise ValueError("descriptor strategy options must be JSON") from error
        for index, record in enumerate(records):
            if record["phase"] != "train":
                raise ValueError("sequence identity requires train phase descriptors")
            if record["content_sha256"] != first["content_sha256"]:
                raise ValueError("descriptor store content identity mismatch")
            if record["active_split_tag"] != first["active_split_tag"]:
                raise ValueError("descriptor active split identity mismatch")
            if record["strategy"] != first["strategy"]:
                raise ValueError("descriptor strategy type mismatch")
            if record["strategy_options_json"] != first["strategy_options_json"]:
                raise ValueError("descriptor strategy configuration mismatch")
            if _integer(record["batch_ordinal"], "batch_ordinal") != index:
                raise ValueError("train phase descriptor order requires contiguous batch ordinals")
        config = _canonical_items(strategy_config, "sampling_strategy_config")
        sampler = _canonical_items(sampler_state, "sampler_rng_identity")
        strategy_type = _nonempty(first["strategy"], "sampling_strategy_type")
        order = tuple(_descriptor_fingerprint(descriptor) for descriptor in values)
        return cls(
            store_content_sha256=_sha256(first["content_sha256"], "store_content_sha256"),
            partition_book_identity=partition_book_identity,
            active_split_tag=_nonempty(first["active_split_tag"], "active_split_tag"),
            fitted_transform_state_key=fitted_transform_state_key,
            sampling_strategy_type=strategy_type,
            sampling_strategy_config=config,
            sampling_strategy_fingerprint=_digest({"config": config, "type": strategy_type}),
            phase_descriptor_digest=_digest(order),
            phase_descriptor_order=order,
            sampler_rng_identity=sampler,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "store_content_sha256": self.store_content_sha256,
            "partition_book_identity": self.partition_book_identity,
            "active_split_tag": self.active_split_tag,
            "fitted_transform_state_key": self.fitted_transform_state_key,
            "sampling_strategy_type": self.sampling_strategy_type,
            "sampling_strategy_config": self.sampling_strategy_config,
            "sampling_strategy_fingerprint": self.sampling_strategy_fingerprint,
            "phase_descriptor_digest": self.phase_descriptor_digest,
            "phase_descriptor_order": self.phase_descriptor_order,
            "sampler_rng_identity": self.sampler_rng_identity,
        }

    @classmethod
    def from_state_dict(cls, state_dict: object) -> "SequenceIdentity":
        state = _exact_keys(state_dict, _IDENTITY_KEYS, "sequence identity")
        config, sampler, order = state["sampling_strategy_config"], state["sampler_rng_identity"], state["phase_descriptor_order"]
        if not isinstance(config, tuple):
            raise TypeError("sampling_strategy_config must be a tuple")
        if not isinstance(sampler, tuple):
            raise TypeError("sampler_rng_identity must be a tuple")
        if not isinstance(order, tuple):
            raise TypeError("phase_descriptor_order must be a tuple")
        fitted = state["fitted_transform_state_key"]
        if fitted is not None and not isinstance(fitted, str):
            raise TypeError("fitted_transform_state_key must be a string or None")
        return cls(
            format_version=state["format_version"], store_content_sha256=state["store_content_sha256"],
            partition_book_identity=state["partition_book_identity"], active_split_tag=state["active_split_tag"],
            fitted_transform_state_key=fitted, sampling_strategy_type=state["sampling_strategy_type"],
            sampling_strategy_config=config, sampling_strategy_fingerprint=state["sampling_strategy_fingerprint"],
            phase_descriptor_digest=state["phase_descriptor_digest"], phase_descriptor_order=order,
            sampler_rng_identity=sampler,
        )


class SequenceState:
    """Ordered transient lifecycle plus one atomic durable commit boundary."""

    def __init__(self, identity: SequenceIdentity, descriptors: Sequence[object]) -> None:
        if not isinstance(identity, SequenceIdentity):
            raise TypeError("identity must be SequenceIdentity")
        values = tuple(descriptors)
        rebuilt = SequenceIdentity.from_descriptors(
            values, partition_book_identity=identity.partition_book_identity,
            fitted_transform_state_key=identity.fitted_transform_state_key,
            sampler_state=dict(identity.sampler_rng_identity),
        )
        if rebuilt != identity:
            raise ValueError("descriptor schedule does not match sequence identity")
        self.identity, self._descriptors = identity, values
        self._issued: dict[int, object] = {}
        self._prepared: set[int] = set()
        self._delivered: set[int] = set()
        self._consumed: set[int] = set()
        self._pending: list[int] = []
        self.committed_cursor = 0
        self.committed_rng_state = self._rng_boundary(0)
        self.committed_sampler_state = self._sampler_boundary(0)
        self.committed_evaluator_sequence = 0
        self.committed_evaluator_count = 0
        self.committed_evaluator_state: object = {}
        self.committed_global_step = 0
        self.committed_epoch = 0
        self._next_delivery = self._next_consume = 1

    @property
    def issued(self) -> tuple[int, ...]: return tuple(sorted(self._issued))
    @property
    def prepared(self) -> tuple[int, ...]: return tuple(sorted(self._prepared))
    @property
    def delivered(self) -> tuple[int, ...]: return tuple(sorted(self._delivered))
    @property
    def consumed(self) -> tuple[int, ...]: return tuple(sorted(self._consumed))
    @property
    def pending_group(self) -> tuple[int, ...]: return tuple(self._pending)
    @property
    def descriptor_count(self) -> int: return len(self._descriptors)
    @property
    def next_issue_id(self) -> int: return max(self._issued, default=self.committed_cursor) + 1

    def descriptor_for(self, sequence_id: int) -> object:
        sequence_id = _integer(sequence_id, "sequence_id", minimum=1)
        return self._descriptors[(sequence_id - 1) % len(self._descriptors)]

    def issue(self, descriptor: object) -> int:
        record = _descriptor_record(descriptor)
        if record["phase"] != "train":
            raise ValueError("only a train phase descriptor can enter sequence state")
        if any(value == descriptor for value in self._issued.values()):
            raise ValueError("descriptor is already issued")
        sequence_id = self.next_issue_id
        expected = self.descriptor_for(sequence_id)
        if descriptor != expected:
            raise ValueError(
                f"descriptor mismatch for expected sequence {sequence_id}: expected fingerprint "
                f"{_descriptor_fingerprint(expected)!r}, received {_descriptor_fingerprint(descriptor)!r}"
            )
        self._issued[sequence_id] = descriptor
        return sequence_id

    def prepare(self, sequence_id: int, descriptor: object) -> None:
        sequence_id = _integer(sequence_id, "sequence_id", minimum=1)
        if sequence_id in self._prepared:
            raise ValueError(f"sequence {sequence_id} is already prepared")
        expected = self._issued.get(sequence_id)
        if expected is None:
            raise ValueError(f"sequence {sequence_id} was not issued")
        if descriptor != expected:
            raise ValueError(f"prepared descriptor mismatch for sequence {sequence_id}")
        self._prepared.add(sequence_id)

    def deliver(self, sequence_id: int) -> object:
        sequence_id = _integer(sequence_id, "sequence_id", minimum=1)
        if sequence_id != self._next_delivery:
            raise ValueError(f"delivery expected sequence {self._next_delivery}, received {sequence_id}")
        if sequence_id not in self._prepared:
            raise ValueError(f"sequence {sequence_id} is not prepared")
        self._delivered.add(sequence_id)
        self._next_delivery += 1
        return self._issued[sequence_id]

    def consume(self, sequence_id: int) -> None:
        sequence_id = _integer(sequence_id, "sequence_id", minimum=1)
        if sequence_id in self._consumed:
            raise ValueError(f"sequence {sequence_id} is already consumed")
        if sequence_id != self._next_consume:
            raise ValueError(f"consumption expected sequence {self._next_consume}, received {sequence_id}")
        if sequence_id not in self._delivered:
            raise ValueError(f"sequence {sequence_id} was not delivered")
        self._consumed.add(sequence_id)
        self._pending.append(sequence_id)
        self._next_consume += 1

    def commit(
        self, *, optimizer_succeeded: bool, model_global_step: int,
        evaluator_sequence: int, evaluator_count: int, evaluator_state: object, epoch: int,
    ) -> bool:
        if type(optimizer_succeeded) is not bool:
            raise TypeError("optimizer_succeeded must be bool")
        model_global_step = _integer(model_global_step, "model_global_step")
        if not optimizer_succeeded or model_global_step == self.committed_global_step:
            return False
        if model_global_step < self.committed_global_step:
            raise ValueError(
                "model global_step regressed below the committed boundary"
            )
        if not self._pending:
            raise ValueError("optimizer commit requires a non-empty pending group")
        expected_group = tuple(range(self.committed_cursor + 1, self._pending[-1] + 1))
        if tuple(self._pending) != expected_group:
            raise ValueError(f"pending sequence group is not contiguous: {self._pending!r}")
        new_cursor = self._pending[-1]
        evaluator_sequence = _integer(evaluator_sequence, "evaluator_sequence")
        if evaluator_sequence != new_cursor:
            raise ValueError(f"evaluator sequence must equal pending cursor {new_cursor}, received {evaluator_sequence}")
        evaluator_count = _integer(evaluator_count, "evaluator_count")
        expected_count = self.committed_evaluator_count + len(self._pending)
        if evaluator_count != expected_count:
            raise ValueError(f"evaluator count expected {expected_count}, received {evaluator_count}")
        epoch = _integer(epoch, "epoch")
        expected_epoch = (new_cursor - 1) // len(self._descriptors)
        if epoch != expected_epoch:
            raise ValueError(f"sequence cursor {new_cursor} belongs to epoch {expected_epoch}, received epoch {epoch}")
        safe_evaluator = _clone_checkpoint_value(evaluator_state, "committed_evaluator_state")
        rng_boundary, sampler_boundary = self._rng_boundary(new_cursor), self._sampler_boundary(new_cursor)
        self.committed_cursor, self.committed_rng_state = new_cursor, rng_boundary
        self.committed_sampler_state = sampler_boundary
        self.committed_evaluator_sequence, self.committed_evaluator_count = evaluator_sequence, evaluator_count
        self.committed_evaluator_state, self.committed_global_step, self.committed_epoch = safe_evaluator, model_global_step, epoch
        for sequence_id in tuple(self._issued):
            if sequence_id <= new_cursor:
                self._issued.pop(sequence_id)
                self._prepared.discard(sequence_id)
                self._delivered.discard(sequence_id)
                self._consumed.discard(sequence_id)
        self._pending.clear()
        return True

    def _rng_boundary(self, cursor: int) -> dict[str, object]:
        if cursor == 0:
            return {"boundary": "initial"}
        descriptor = self.descriptor_for(cursor)
        return {
            "generator_seed": _integer(getattr(descriptor, "generator_seed"), "generator_seed"),
            "generator_state_sha256": _sha256(getattr(descriptor, "generator_state_sha256"), "generator_state_sha256"),
        }

    def _sampler_boundary(self, cursor: int) -> dict[str, object]:
        return {"cursor": cursor, "identity": self.identity.sampler_rng_identity}

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": _STATE_FORMAT, "identity": self.identity.state_dict(),
            "committed_cursor": self.committed_cursor,
            "committed_rng_state": _clone_checkpoint_value(self.committed_rng_state, "committed_rng_state"),
            "committed_sampler_state": _clone_checkpoint_value(self.committed_sampler_state, "committed_sampler_state"),
            "committed_evaluator_sequence": self.committed_evaluator_sequence,
            "committed_evaluator_count": self.committed_evaluator_count,
            "committed_evaluator_state": _clone_checkpoint_value(self.committed_evaluator_state, "committed_evaluator_state"),
            "committed_global_step": self.committed_global_step, "committed_epoch": self.committed_epoch,
        }

    def load_state_dict(self, state_dict: object) -> None:
        state = _exact_keys(state_dict, _STATE_KEYS, "sequence state")
        if state["format_version"] != _STATE_FORMAT:
            raise ValueError(f"unsupported sequence state version {state['format_version']!r}")
        if SequenceIdentity.from_state_dict(state["identity"]) != self.identity:
            raise ValueError("sequence identity mismatch while restoring state")
        cursor = _integer(state["committed_cursor"], "committed_cursor")
        evaluator_sequence = _integer(state["committed_evaluator_sequence"], "committed_evaluator_sequence")
        evaluator_count = _integer(state["committed_evaluator_count"], "committed_evaluator_count")
        if evaluator_sequence != cursor or evaluator_count != cursor:
            raise ValueError("committed evaluator sequence/count must equal committed cursor")
        global_step = _integer(state["committed_global_step"], "committed_global_step")
        if (cursor == 0 and global_step != 0) or (
            cursor > 0 and global_step < 1
        ):
            raise ValueError("committed global_step is outside the cursor boundary")
        epoch = _integer(state["committed_epoch"], "committed_epoch")
        expected_epoch = 0 if cursor == 0 else (cursor - 1) // len(self._descriptors)
        if epoch != expected_epoch:
            raise ValueError(f"committed epoch expected {expected_epoch}, received {epoch}")
        rng_state = _clone_checkpoint_value(state["committed_rng_state"], "committed_rng_state")
        if rng_state != self._rng_boundary(cursor):
            raise ValueError("committed RNG boundary does not match cursor")
        sampler_state = _clone_checkpoint_value(state["committed_sampler_state"], "committed_sampler_state")
        if sampler_state != self._sampler_boundary(cursor):
            raise ValueError("committed sampler boundary does not match cursor")
        evaluator_state = _clone_checkpoint_value(state["committed_evaluator_state"], "committed_evaluator_state")
        self.committed_cursor, self.committed_rng_state, self.committed_sampler_state = cursor, rng_state, sampler_state
        self.committed_evaluator_sequence, self.committed_evaluator_count = evaluator_sequence, evaluator_count
        self.committed_evaluator_state, self.committed_global_step, self.committed_epoch = evaluator_state, global_step, epoch
        self._issued.clear(); self._prepared.clear(); self._delivered.clear(); self._consumed.clear(); self._pending.clear()
        self._next_delivery = self._next_consume = cursor + 1


__all__ = ["SequenceIdentity", "SequenceState"]
