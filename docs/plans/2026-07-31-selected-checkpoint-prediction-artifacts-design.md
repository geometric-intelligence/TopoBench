# Selected-Checkpoint Prediction Artifacts Design

## Status

Approved on 2026-07-31 for integration into the graph, heterogeneous graph,
and hypergraph core remediation plan.

## Objective

Make the validation and test evaluations of the validation-selected checkpoint
auditable. Each run must retain split-specific scalar metrics, the exact
`num_examples` that participated in those metrics, plus per-sample identity,
target, raw model output, exported prediction, optional normalized or
inverse-transformed values, and explicitly allowlisted lightweight metadata.

The implementation extends TopoBench's existing `TBModel`, supervision,
callback, logger, and selected-checkpoint rerun lifecycle. It is not a second
evaluation framework and does not introduce domain-specific training loops.

## Authoritative Output Layout

Each Hydra run directory owns exactly two final artifact roots:

```text
evaluations/
  best_checkpoint/
    val/
      metrics.json
      predictions/
        manifest.json
        part-00000.npz
        part-00001.npz
        ...
    test/
      metrics.json
      predictions/
        manifest.json
        part-00000.npz
        part-00001.npz
        ...
```

The fully resolved roots are `evaluations/best_checkpoint/val` and
`evaluations/best_checkpoint/test`; they are never aliases for one shared
prediction directory.

`val` and `test` are independent artifacts. Neither split may reuse a file,
manifest, logger artifact name, temporary directory, or mutable writer state
owned by the other split. The validation rerun and test rerun both load the
same checkpoint selected solely by validation. Test metrics never influence
checkpoint selection.

The local run directory is authoritative. Logger uploads mirror these files;
they do not become the only retained copy.

## TopoBench Integration Boundary

The extension uses four small contracts behind existing framework seams:

1. The canonical `EvaluationBatch` carries the loss-owning logits, targets,
   exact example count, and `PredictionIdentity` aligned row-for-row. The
   writer never introduces a second supervision/evaluator batch type.
2. The configured data pipeline supplies a `PredictionRowAdapter`. It attaches
   canonical IDs before batching and converts the selected supervision rows to
   one domain-neutral prediction payload. Domain logic does not branch in
   `run.py` or in the artifact writer.
3. `SelectedCheckpointArtifactCallback` writes only while the existing
   validation-selected checkpoint reruns execute. Ordinary training,
   validation during fitting, and ad-hoc test calls do not produce final
   prediction artifacts.
4. `ArtifactLoggerAdapter` registers each completed local file with every
   configured logger using stable split-qualified names.

`TBModel._model_step` remains the single loss/evaluator boundary. It returns a
small standard prediction payload only when artifact capture is enabled. The
payload references the already selected logits and targets; it does not run a
second forward pass or repeat supervision selection.

## Stable Identity Contract

Every row has a canonical entity identity and split-local ordinal. IDs are
assigned or resolved by the data pipeline, never inferred from batch position
inside the writer.

- Graph-level homogeneous prediction: `sample_id`; use a declared external
  graph/example ID when available, otherwise a deterministic dataset ordinal.
  `source_graph_id` and `source` may be separate allowlisted metadata fields.
- Homogeneous node prediction: `(source_graph_id, global_nid)`. Native
  full-graph data receives the same canonical node ordinals used by the
  disk-backed view. Disk batches already carry canonical `global_nid`.
- Heterogeneous node prediction:
  `(source_graph_id, target_node_type, n_id)`. In neighbor mode only
  `n_id[:batch_size]` target seeds are exported; sampled context nodes never
  become prediction rows.
- Hypergraph node prediction: `(source_graph_id, global_nid)`.

External integer or string IDs are restored only at export when a validated
ID resolver is present. They never need to enter GPU model batches. Composite
identity prevents collisions when the same external ID occurs in multiple node
types or source graphs.

A completed split must contain each expected supervised identity exactly once.
Duplicate, missing, reordered-without-ordinal, or surplus rows fail artifact
promotion. The manifest records the expected and observed row counts and the
identity-column schema.

## Per-Row Payload

Every shard stores column-oriented arrays with equal first dimension:

- canonical identity columns and `split_ordinal`;
- `target` in exported task semantics;
- `raw_output`, exactly the model output selected for loss before probability
  conversion or inverse target transformation;
- `prediction`, the evaluator-facing value: probabilities or class score
  representation for classification, and configured export units for
  regression;
- optional `target_model_space`, `prediction_model_space`,
  `target_normalized`, or `prediction_normalized` only when the active target
  transform defines them;
- allowlisted lightweight metadata such as `source`, acquisition group,
  cohort, or sensor identifier.

The manifest defines every column's semantic role, shape tail, NumPy dtype,
units or class vocabulary where applicable, and transformation fingerprint.
Binary, multiclass, multilabel, and regression outputs must have explicit,
non-broadcasting shapes. No field may contain an arbitrary Python object.

Metadata is configured as semantic field names supplied by the data pipeline.
Arbitrary Python attribute paths, SQL, callbacks, tensors unrelated to the
selected sample, whole sampled neighborhoods, embeddings, or raw inputs are
prohibited. High-cardinality metadata is permitted in shards but never
expanded into unbounded scalar logger keys.

## Format and Sharding

Core output uses versioned JSON plus uncompressed NumPy `.npz` shards written
with `allow_pickle=False`. This keeps the default core free of a PyArrow
runtime dependency while supporting numeric, boolean, and fixed-width Unicode
columns. Parquet export may be added later as an optional adapter; it is not a
second canonical format.

The writer is streaming and byte-bounded:

- copy one completed evaluation batch to CPU;
- append columns to a bounded row/byte buffer;
- flush `part-NNNNN.npz` in canonical evaluation order;
- release the buffer before accepting more rows;
- never materialize all split predictions in RAM.

Shard row and byte caps are explicit configuration fields and appear in the
manifest. A very large single row fails with a precise size error rather than
silently violating the memory bound.

## Manifest and Metrics Schemas

`predictions/manifest.json` is versioned and contains:

- artifact schema version, split, task, task level, domain, target node type
  where relevant, and output semantics;
- selected epoch/global step, checkpoint path relative to the run root, and
  checkpoint SHA-256;
- dataset/source/store, active split tag and three phase-source digests,
  accepted partition book, fitted/runtime transform state, model, environment,
  checkpoint, and run-provenance/reproducibility-bundle fingerprints;
- ordered column schema and identity key;
- expected/observed row count, canonical order policy, and uniqueness result;
- shard names, row ranges, byte sizes, and SHA-256 digests;
- creation status/timestamp and writer configuration;
- logger registration names and status, without credentials.

`metrics.json` is a versioned object containing split/checkpoint identity,
integer `num_examples`, ordered scalar `metrics`, and bounded metric metadata
including exactness, thresholds, class support, undefined reasons, and
resource evidence. `num_examples` comes directly from the authoritative
`EvaluationResult`; it is the number of supervised datapoints that participated
in every configured metric. It must equal the prediction manifest's observed
row count before promotion. The file is generated from the same evaluator
lifecycle as returned metrics and logger values, not recomputed from a separate
implementation.

Configured evaluation slices such as `source` use a declared finite vocabulary,
a minimum row count, and a maximum category count. Their metrics are written
under a stable `slices/<field>/<value>/...` structure and logged with the same
namespace. Unknown or excessive categories fail preflight or remain only as
per-row metadata according to configuration; they never create unbounded
logger cardinality.

## Atomicity, Collision, and Resume

Each split writes to a locked temporary sibling directory. Promotion occurs
only after row coverage, identity uniqueness, shapes, finiteness, shard
checksums, manifest, and metrics checks pass. The completed directory is then
renamed atomically.

Final files are immutable. A writer never silently overwrites them. Repeating
the exact same split/checkpoint evaluation is an idempotent success only when
all existing digests and fingerprints match. Any mismatch fails and preserves
both the existing artifact and diagnostic staging state.

For partitioned runs, idempotence also requires the same content-addressed
store, accepted partition map, active split tag, fitted-transform state, and
declared reproduction profile. A moved but checksum-identical store remains the
same identity; rerunning METIS or silently refitting preprocessing does not.

Prediction-writer progress is not mixed with training sampler state. A
checkpoint taken during training contains no partial final prediction arrays.
If a selected-checkpoint rerun is interrupted, it restarts that split from its
canonical beginning or resumes only from a validated shard boundary with the
same checkpoint, split, and data fingerprints. Final publication remains
all-or-nothing.

Distributed prediction export is outside the qualified scope. If world size is
greater than one and no explicit all-rank merge adapter is configured, startup
fails rather than allowing rank zero to publish incomplete predictions.

## Logger Contract

Every completed file is registered separately with every configured experiment
logger:

- `best-checkpoint-val-metrics`;
- `best-checkpoint-val-predictions-manifest`;
- `best-checkpoint-val-predictions-part-00000`, and so on;
- corresponding distinct `best-checkpoint-test-*` names.

The logger bridge is explicit and capability-checked. The W&B adapter creates
one immutable artifact per local file with digest, split, checkpoint, and run
metadata. The CSV/local adapter writes one append-only artifact-index record
per file containing its URI and SHA-256. A configured logger without a
supported artifact adapter causes preflight failure when prediction artifacts
are enabled; files are never silently omitted from one logger.

Scalar metrics and integer `num_examples` are logged separately under
`evaluations/best_checkpoint/{val,test}/...` to every logger. The count is
logged once from the finalized result, never as an average of batch counts.
Artifact metadata must not be flattened into scalar metric namespaces.

## Configuration Contract

```yaml
evaluation_artifacts:
  enabled: true
  root: ${hydra:runtime.output_dir}/evaluations/best_checkpoint
  splits: [val, test]
  prediction_format: npz
  shard_rows: 65536
  shard_bytes_mb: 256
  metadata_fields: [source]
  evaluation_slices:
    source:
      max_categories: 64
      min_rows: 1
  distributed_policy: reject
  existing_artifact_policy: verify_identical
```

Packaged qualified runs enable both splits. Configuration validation rejects
unsupported formats, duplicate roots, unsafe paths outside the run directory,
non-positive bounds, unknown semantic metadata fields, or a logger lacking an
artifact adapter.

## Error Handling

Fail before final promotion for:

- no validation-selected checkpoint, unreadable checkpoint, or digest change
  between the two reruns;
- target/logit/prediction row or shape mismatch;
- missing, duplicate, or surplus identities;
- disagreement among evaluator `num_examples`, `metrics.json`, logger value,
  and manifest observed row count;
- heterogeneous context nodes exported as seeds;
- absent graph-level sample IDs after pipeline normalization;
- non-finite required outputs unless the task contract explicitly permits
  them;
- inconsistent metadata width/type or undeclared metadata;
- shard or manifest checksum mismatch;
- val/test path or logger-name collision;
- unsupported logger artifact capability;
- partial multi-rank output.

A logger upload failure marks the run unsuccessful and records which local
artifact remains available for retry. It never deletes or invalidates the
already promoted local copy.

## Qualification

The implementation must prove:

1. selected-checkpoint validation and test reruns each execute once and use the
   same validation-selected checkpoint digest;
2. their metrics, counts, directories, manifests, shards, and logger artifact
   names are distinct;
3. graph-level batches retain stable example IDs across shuffle and batching;
4. homogeneous full-graph and disk-cluster node outputs use canonical global
   IDs and restore external IDs exactly;
5. heterogeneous neighbor output contains target seeds only and allows the
   same external ID in two node types without collision;
6. hypergraph node output aligns IDs, targets, and predictions;
7. classification logits/probabilities and regression model/export-space
   values obey declared shapes and transformations;
8. per-source metadata and bounded source-sliced metrics align with the same
   rows and report their own participant counts;
9. multi-shard output stays within configured memory bounds and round-trips
   with `allow_pickle=False`;
10. `EvaluationResult.num_examples`, `metrics.json`, the manifest observed row
    count, returned results, provenance, and each logger value agree exactly;
11. interruption cannot expose a partial final artifact; exact rerun is
    idempotent and a conflicting rerun is rejected;
12. every configured logger receives a separate record/upload for every file;
13. `run()` returns final metrics plus val/test artifact manifest paths and
    digests, and immutable run provenance plus the mandatory qualified
    reproducibility bundle index the same artifacts;
14. fresh/cache/download/moved-store/resumed evaluations preserve active split,
    partition, transform, checkpoint, row identity, metrics, and artifact
    digests under the declared exact or numeric-equivalence profile.

## Non-goals

- Training-epoch prediction retention.
- Saving inputs, embeddings, sampled neighborhoods, or arbitrary batch state.
- A general dataframe/reporting framework.
- Parquet as a mandatory prediction dependency.
- Distributed all-rank merge in the first qualified release.
- Test-driven checkpoint selection.
- Replacing Lightning loggers, `TBModel`, supervision adapters, or TopoBench's
  data-pipeline architecture.
