# Disk-backed Key Index (SQLite)

Overview
- Purpose: Provide an optional, production-grade, disk-backed key map for the key-driven pipeline.
- Motivation: Large key tables (millions of rows) cannot be safely loaded into memory. A SQLite-backed index
  keeps memory stable, allows O(1)-ish lookups by filename, and can be reused across multiple runs.
- Backward compatibility: Disabled by default. When `key_input.disk_backed_index.enabled=false`, behavior is identical
  to the legacy in-memory path, including Part 2 optimizations.

When to enable
- You process very large key tables and encounter RAM spikes.
- You want to reuse the key index across runs and avoid reloading/reparsing on each execution.
- You need case-insensitive lookups to prevent key mismatches due to filename casing.

High-level design
- A SQLite database stores a minimal schema:
  - `key TEXT NOT NULL UNIQUE`
  - `normalized_key TEXT` (optional when `case_insensitive_keys=true`)
  - A column for each required data field (see Required Columns)
  - Optional `row_json TEXT` when `store_all_columns=true`
- A `metadata_table` records source signature and schema so the orchestrator can decide whether to rebuild.
- Ingestion is streaming and committed in chunks (`index_chunk_size`) for throughput and memory stability.
- Casting is applied at ingestion using `sqlite.casting` rules; type hints live under `sqlite.column_types`.
- Runtime lookups use indexed key columns and return a `{ filename -> row }` map for the current batch.

Required Columns
- Always include the filename column: `key_input.file_name_column`.
- Union of:
  - `data_source.api.request_mapping.param_map` KEYS (the key table columns you map to API params), and
  - `key_input.columns_for_results` (for enrichment in results)
- If `store_all_columns=true`, the full source row is also stored as JSON for debugging/forensics but is not
  required for runtime lookups.

Configuration (YAML)
- Added under `key_input.disk_backed_index`:
  - `enabled`: Master switch. Default `false` for safety. When `false`, nothing changes.
  - `backend`: Currently `sqlite`. Reserved for future backends (e.g., LMDB).
  - `persist_disk_backed_index`: When `true`, persist the SQLite DB across runs. When `false`, create a fresh temp DB per run and delete it after completion (ephemeral mode). Default `true`.
  - `index_chunk_size`: Number of rows per transaction during ingestion. Defaults to `10000`.
  - `store_all_columns`: When `true`, also persist the entire row as `row_json`.
  - `case_insensitive_keys`: When `true`, also store and index `normalized_key = lower(filename)`.
  - `rebuild_policy`:
    - `on_schema_change`: When `true`, rebuild if required columns/types or `file_name_column`/`case_insensitive_keys` change.
    - `on_source_change`: `mtime_only` | `mtime_hash` | `never`. Prefer `mtime_hash` for safety.
    - `force_rebuild`: Force rebuild regardless of checks.
  - `sqlite`:
    - `db_path`: SQLite DB location. Relative to repo root unless absolute.
    - `table_name`, `metadata_table`: Table names. Defaults are `key_index`, `key_index_meta`.
    - PRAGMAs: `journal_mode` and `synchronous` to tune durability/performance. Defaults: `WAL`, `NORMAL`.
    - `strict_mode`: Try to create a STRICT table when supported (SQLite 3.37+). Fallback silently if unsupported.
    - `default_type`: Default column type for required columns without explicit mapping.
    - `column_types`: Per-column type hints influencing schema and default outbound casting.
    - `casting`: Per-column ingestion casts (e.g., `int`, `float`, `date_to_iso`, `bool`).
    - `on_cast_error`: `store_text` (default) | `store_null` | `raise`.
    - `date_policy`: Default date-like handling (`iso_text` recommended). Can be overridden per-request at runtime.

Rebuild Policy
- The orchestrator calls `is_index_up_to_date()` to decide whether to rebuild.
- Reasons to rebuild:
  - DB missing
  - Metadata or schema mismatch (limited by `on_schema_change`)
  - Source changes detected per `on_source_change`
  - `force_rebuild=true`
- `on_source_change=never` disables source signature checks; use with caution if the input file can change.

Persistence and Ephemeral Mode
- `persist_disk_backed_index=true` (default): The DB at `sqlite.db_path` is reused across runs if up-to-date.
- `persist_disk_backed_index=false`: The orchestrator creates a unique temp DB path per run, builds the index as needed, and deletes the DB (and its containing temp folder) at the end of the run.
- Ephemeral mode is useful for one-off runs or when you do not want state persisted between executions.

Runtime Behavior
- When enabled, the orchestrator avoids building an in-memory `rows_map` for large runs.
- For each batch:
  - API mode: performs a DB lookup for the batch and builds the request payloads.
  - Local mode with enrichment: performs a DB lookup and propagates only the requested `columns_for_results`.
- Missing rows are logged; the batch still proceeds.

Outbound Casting
- Request-time casting precedence in `ApiFetcher`:
  1) `data_source.api.request_mapping.param_cast[api_param]` (explicit override)
  2) SQLite `column_types` (used as default hints when disk index is enabled)
  3) Passthrough
- Supported outbound types: `string`, `int`, `float`, `bool`, `datetime`.
  - `datetime` supports `input_format`, `as` (`iso`, `epoch_seconds`, `epoch_millis`) and `tz` (currently normalized to UTC naive).
  - `on_error: null` sends `null` when casting fails; omit or set to other values to keep the original.

Tuning Guidance
- `index_chunk_size`:
  - Too small -> many commits, lower throughput.
  - Too large -> large transactions; if in doubt, start with 10k.
- PRAGMAs (`WAL` + `NORMAL`): good balance for concurrent reads and durability.
- `case_insensitive_keys=true` recommended to avoid casing issues.
- `store_all_columns=false` recommended for performance unless you need full row JSON for audits.

Error Handling and Safety
- Ingestion:
  - Missing required columns -> fail fast with a descriptive error listing missing columns.
  - Casting failures follow `on_cast_error` and do not crash builds by default.
- Lookups:
  - Missing row for a key -> warn and continue.
- Backward Compatibility:
  - Disabled by default; same behavior as in-memory path.

Testing Outline
- Unit tests:
  - Required column derivation and declared types from YAML
  - Rebuild detection on schema/source change and forced rebuild
  - Ingestion casting and error policy
  - Lookups with `case_insensitive_keys`
  - Payload casting precedence
- Integration tests:
  - Large synthetic CSV (repeat rows to simulate 1M+) builds with minimal memory usage, commits every 10k
  - API mode with `enabled=true` yields correctly typed payloads; enrichment works as configured
  - Disabled path identical to current system; search_results_summary.json produced
  - Performance logs: rows/sec during build; lookup latency (batch sizes 100, 1000, 5000)

Operational Notes
- The DB is stored under `instance/key_table_index.sqlite` by default. Back it up when needed.
- If schema or source changes unexpectedly, set `force_rebuild: true` for a clean rebuild.
- Consider placing `instance` on a fast local disk (SSD/NVMe) for best performance.

Maintainer Notes (future-proofing)
- The module `sqlite_key_index.py` was written to be backend-agnostic at the interface level. To add LMDB:
  - Implement a parallel module with the same public functions.
  - Extend the orchestrator to switch by `backend`.
  - Reuse the derivation and casting logic as-is.
