"""
TursoVectorStore — VectorStore implementation backed by Turso / libSQL.

Each "collection" maps to a per-collection table in libSQL:
  vec_{_s(name)} — stores dense VECTOR(dim) + payload JSON
  vec_{_s(name)}_fts — FTS5 sidecar for sparse/lexical search (via FTS5 on text_content)

Hybrid query: run query_dense + query_sparse, fuse with Python RRF.

Phase 3 of the backend abstraction plan.

S2 note: libSQL accepts vectors with or without spaces, so json.dumps works.
S3 note: query_sparse requires FTS query syntax — see _sparse_to_fts_query.
S4 note: bm25() returns negative scores — we negate so ScoredPoint.score is "higher = better" (S5 contract).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Iterator, Iterable, Any

from shared.store_models import (
    PointStruct, ScoredPoint, Filter, FieldCondition,
    MatchValue, MatchText, MatchAny, MatchContains, Range,
    CollectionInfo, SparseVector,
)  # noqa: F401  (MatchText exported for completeness)
from shared.text_search_utils import escape_fts5_query as _escape_fts5_query

logger = logging.getLogger(__name__)

# Keys that are safe to interpolate into a JSON path ($.key) for json_set.
# Anything outside [A-Za-z_$][A-Za-z0-9_$]* can't be a bare path member and falls
# back to json_patch in set_payload. Only alphanumerics/underscore/dollar → no SQL
# or path injection is possible when interpolating into the statement.
_BAREWORD_KEY = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")


def _serialize_vector(vec: list[float]) -> str:
    """Serialize a vector for libSQL's VECTOR type (S2)."""
    return "[" + ",".join(repr(x) for x in vec) + "]"


def _s(name: str) -> str:
    """Sanitize collection name for use as SQL identifier (handles hyphens, dots)."""
    return name.replace("-", "_").replace(".", "_").replace("/", "_")


def _rrf_fuse(dense_hits: list[ScoredPoint], sparse_hits: list[ScoredPoint],
              limit: int = 10, k: int = 60) -> list[ScoredPoint]:
    """Reciprocal Rank Fusion — Python-side for hybrid queries."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    for rank, hit in enumerate(dense_hits):
        scores[str(hit.id)] = scores.get(str(hit.id), 0) + 1.0 / (k + rank + 1)
        payloads[str(hit.id)] = hit.payload
    for rank, hit in enumerate(sparse_hits):
        scores[str(hit.id)] = scores.get(str(hit.id), 0) + 1.0 / (k + rank + 1)
        payloads[str(hit.id)] = hit.payload
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:limit]
    return [ScoredPoint(id=id, score=score, payload=payloads.get(id))
            for id, score in ranked]


class TursoVectorStore:
    """VectorStore backed by Turso / libSQL."""

    def __init__(self, url: str, auth_token: str | None = None):
        import libsql_experimental as libsql

        # Concurrency: single shared connection is safe — libsql_experimental's
        # C binding serializes statements via an internal mutex (stress-tested:
        # 4 threads x 50 ops = 200/200 correct, no lock errors). autocommit=True
        # keeps statements independent so threads don't block on implicit txns.
        # No threading.Lock needed (would only reduce throughput); see the note
        # in TursoSqlStore.__init__ for the full rationale.
        if auth_token:
            self._conn = libsql.connect(url, auth_token=auth_token)
        else:
            self._conn = libsql.connect(url)
        self._conn.autocommit = True
        # Detect vector support
        self._has_vector = self._detect_vector_support()
        self._has_hnsw = self._detect_hnsw()
        if not self._has_vector:
            logger.warning("libSQL VECTOR type not available — vector queries will fail")
        elif not self._has_hnsw:
            logger.warning("libSQL HNSW not available — vector queries will be O(N) brute-force")
        logger.info(f"TursoVectorStore initialized (url={url[:30]}..., vector={self._has_vector}, hnsw={self._has_hnsw})")

    def _detect_vector_support(self) -> bool:
        """Check if VECTOR column type is supported."""
        try:
            self._conn.execute("CREATE TABLE _probe_v (v VECTOR(1))")
            self._conn.execute("DROP TABLE _probe_v")
            return True
        except Exception:
            return False

    def _detect_hnsw(self) -> bool:
        """Check if HNSW index is supported."""
        if not self._has_vector:
            return False
        try:
            self._conn.execute("CREATE TABLE _probe_hnsw (v VECTOR(1))")
            self._conn.execute("CREATE INDEX _test_hnsw ON _probe_hnsw USING hnsw(v)")
            self._conn.execute("DROP TABLE _probe_hnsw")
            return True
        except Exception:
            # Clean up if any leftover
            try:
                self._conn.execute("DROP TABLE IF EXISTS _probe_hnsw")
            except Exception:
                pass
            return False

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def _ensure_map_table(self) -> None:
        """Create the collection-name mapping table (sanitized -> original)."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS _vec_collection_map (
                sanitized TEXT PRIMARY KEY,
                original  TEXT NOT NULL
            )
        """)

    def _store_name_mapping(self, name: str) -> None:
        """Record the original collection name alongside the sanitized one."""
        self._ensure_map_table()
        self._conn.execute(
            "INSERT OR REPLACE INTO _vec_collection_map (sanitized, original) VALUES (?, ?)",
            (_s(name), name),
        )

    def _lookup_original_name(self, sanitized: str) -> str:
        """Reverse-lookup the original collection name from a sanitized table name."""
        self._ensure_map_table()
        row = self._conn.execute(
            "SELECT original FROM _vec_collection_map WHERE sanitized = ?", (sanitized,)
        ).fetchone()
        return row[0] if row else sanitized

    def ensure_collection(
        self, name: str, *,
        dense_dim: int | None = None,
        sparse: bool = False,
        distance: str = "Cosine",
    ) -> None:
        if dense_dim is not None:
            try:
                self._conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS vec_{_s(name)} (
                        id          TEXT PRIMARY KEY,
                        embedding   VECTOR({dense_dim}),
                        payload     TEXT,
                        text_content TEXT,
                        created_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                        updated_at  TEXT
                    )
                """)
            except Exception as e:
                err = str(e).lower()
                if "already exists" not in err:
                    # Fallback without vector type
                    self._conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS vec_{_s(name)} (
                            id          TEXT PRIMARY KEY,
                            embedding   BLOB,
                            payload     TEXT,
                            text_content TEXT
                        )
                    """)
        else:
            # No dense — FTS5-only collection
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS vec_{_s(name)} (
                    id          TEXT PRIMARY KEY,
                    payload     TEXT,
                    text_content TEXT
                )
            """)

        # Record the original -> sanitized mapping for list_collections
        self._store_name_mapping(name)

        if sparse:
            try:
                self._conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_{_s(name)}_fts USING fts5(
                        text_content,
                        content='vec_{_s(name)}',
                        content_rowid='rowid',
                        tokenize='porter unicode61'
                    )
                """)
            except Exception as e:
                logger.debug(f"FTS5 sidecar creation failed: {e}")

        # HNSW index if supported
        if dense_dim is not None and self._has_hnsw:
            try:
                self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{_s(name)}_hnsw ON vec_{_s(name)} USING hnsw(embedding)")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert(self, collection: str, points: list[PointStruct]) -> None:
        # Check if FTS5 sidecar exists
        fts_exists = False
        try:
            fts_check = self._conn.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='vec_{_s(collection)}_fts'"
            ).fetchone()
            fts_exists = bool(fts_check)
        except Exception:
            fts_exists = False

        for p in points:
            payload_str = json.dumps(p.payload or {})
            # Index the searchable text. memorymcp payloads use "text"; ragmcp
            # payloads use "codeChunk". Fall through so the FTS5 sidecar is
            # populated for both tools.
            pl = p.payload or {}
            text_content = pl.get("text") or pl.get("codeChunk") or pl.get("content") or ""
            vec_str = ""
            if p.vector and isinstance(p.vector, list) and len(p.vector) > 0:
                vec_str = _serialize_vector(p.vector)
            elif p.vector and isinstance(p.vector, dict):
                for v in p.vector.values():
                    if isinstance(v, list):
                        vec_str = _serialize_vector(v)
                        break

            self._conn.execute(f"""
                INSERT INTO vec_{_s(collection)} (id, embedding, payload, text_content)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    embedding = excluded.embedding,
                    payload = excluded.payload,
                    text_content = excluded.text_content,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            """, (str(p.id), vec_str, payload_str, text_content))

            # FTS5 sync (only if sidecar exists)
            if fts_exists and text_content:
                fts_row = self._conn.execute(
                    f"SELECT rowid FROM vec_{_s(collection)} WHERE id = ?", (str(p.id),)
                ).fetchone()
                if fts_row:
                    self._conn.execute(
                        f"INSERT INTO vec_{_s(collection)}_fts(rowid, text_content) VALUES (?, ?)",
                        (fts_row[0], text_content),
                    )

    def set_payload(self, collection: str, payload: dict, *, ids: list[str]) -> None:
        """Shallow top-level merge of ``payload`` into each point's payload.

        Matches Qdrant's native ``set_payload`` and PG's jsonb ``||``: keys
        present in ``payload`` overwrite the stored value at that top-level key
        (object values REPLACE wholesale — they are NOT deep-merged), while keys
        absent from ``payload`` are preserved. Callers like queryMemory rely on
        this to update a few fields (e.g. ``{"usage_count": N}``) without
        clobbering the rest.

        Implemented as a single atomic UPDATE. For bareword-safe keys (all real
        payload keys) we use ``json_set`` so each key's value is replaced in
        place — this keeps the merge atomic, so two concurrent cross-key
        ``set_payload`` calls on the same point can't lose a sibling key (a
        Python read-modify-write would). Keys that are not bareword-safe cannot
        be expressed as a JSON path and fall back to ``json_patch`` (atomic, but
        deep-merging — identical to shallow for the flat payloads we store).
        """
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)

        keys = list(payload.keys())
        if keys and all(_BAREWORD_KEY.match(k) for k in keys):
            expr = "payload"
            set_params = []
            for k in keys:
                expr = f"json_set({expr}, '$.{k}', json(?))"
                set_params.append(json.dumps(payload[k]))
            self._conn.execute(
                f"UPDATE vec_{_s(collection)} SET payload = {expr} WHERE id IN ({placeholders})",
                tuple(set_params + ids),
            )
        else:
            # Fallback for non-bareword keys: atomic deep-merge (== shallow for flat payloads)
            self._conn.execute(
                f"UPDATE vec_{_s(collection)} SET payload = json_patch(payload, ?) WHERE id IN ({placeholders})",
                (json.dumps(payload), *ids),
            )

    def delete(self, collection: str, *,
               ids: list[str] | None = None,
               filter: Filter | None = None) -> None:
        if ids:
            placeholders = ",".join("?" for _ in ids)
            self._conn.execute(
                f"DELETE FROM vec_{_s(collection)} WHERE id IN ({placeholders})",
                tuple(ids),
            )
        elif filter:
            where, params = _filter_to_sql(filter, "payload")
            if where:
                self._conn.execute(
                    f"DELETE FROM vec_{_s(collection)} WHERE {where}",
                    tuple(params),
                )

    def delete_collection(self, name: str) -> None:
        self._conn.execute(f"DROP TABLE IF EXISTS vec_{_s(name)}")
        self._conn.execute(f"DROP TABLE IF EXISTS vec_{_s(name)}_fts")

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def query_dense(self, collection: str, vec: list[float], *,
                    limit: int = 10, filter: Filter | None = None,
                    using: str | None = None) -> list[ScoredPoint]:
        vec_str = _serialize_vector(vec)
        where, filter_params = _filter_to_sql(filter, "payload")
        where_clause = f"WHERE {where}" if where else ""
        all_params = [vec_str] + filter_params + [vec_str, limit]
        sql = f"""
                SELECT id, payload, vector_distance_cos(embedding, ?) AS d
                FROM vec_{_s(collection)}
                {where_clause}
                ORDER BY vector_distance_cos(embedding, ?)
                LIMIT ?
            """
        try:
            rows = self._conn.execute(sql, tuple(all_params)).fetchall()
        except Exception as e:
            logger.error(f"query_dense failed: {e}")
            return []
        results = []
        for row in rows:
            payload = json.loads(row[1]) if row[1] else {}
            # S5: ScoredPoint.score is similarity (higher = better); distance is negated
            results.append(ScoredPoint(id=str(row[0]), score=1.0 - row[2], payload=payload))
        return results

    def query_sparse(self, collection: str, sparse: SparseVector, *,
                     limit: int = 10, filter: Filter | None = None,
                     query_text: str | None = None) -> list[ScoredPoint]:
        """Lexical search via the FTS5 sidecar.

        Prefers ``query_text`` (a real FTS5 MATCH against indexed text) since the
        integer ``sparse.indices`` are process-local hashes that cannot match
        lexical tokens. Without ``query_text`` we fall back to the sparse terms,
        which will typically match nothing here — callers should always pass
        ``query_text`` when they have the raw query string.
        """
        fts_table = f"vec_{_s(collection)}_fts"
        base_table = f"vec_{_s(collection)}"

        # Build the FTS5 query string
        if query_text:
            fts_query = _escape_fts5_query(query_text)
        elif sparse.indices:
            # Fallback: integer hashes wrapped as phrases — rarely matches real tokens.
            fts_query = " OR ".join(f'"{i}"' for i in sparse.indices[:10])
        else:
            return []

        where, params = _filter_to_sql(filter, "v.payload")
        where_clause = f"AND {where}" if where else ""
        params = [fts_query] + params + [limit]
        try:
            rows = self._conn.execute(f"""
                SELECT v.id, v.payload, bm25({fts_table}) AS s
                FROM {fts_table}
                JOIN {base_table} v ON v.rowid = {fts_table}.rowid
                WHERE {fts_table} MATCH ?
                {where_clause}
                ORDER BY s
                LIMIT ?
            """, tuple(params)).fetchall()
        except Exception as e:
            logger.error(f"query_sparse failed: {e}")
            return []
        results = []
        for row in rows:
            raw_bm25 = row[2]
            # S4: bm25 returns NEGATIVE scores (lower = better match). Negate so
            # ScoredPoint.score is similarity (higher = better). A future libSQL
            # build returning positive bm25 would invert ranking — warn rather
            # than assert so we still return results.
            if raw_bm25 > 0:
                logger.warning(
                    "bm25 returned positive score (%s); expected <= 0 — ranking may be inverted",
                    raw_bm25,
                )
                score = raw_bm25
            else:
                score = -raw_bm25
            payload = json.loads(row[1]) if row[1] else {}
            results.append(ScoredPoint(id=str(row[0]), score=score, payload=payload))
        return results

    def query_hybrid(self, collection: str, dense: list[float],
                     sparse: SparseVector, *, limit: int = 10,
                     filter: Filter | None = None,
                     query_text: str | None = None) -> list[ScoredPoint]:
        dense_hits = self.query_dense(collection, dense, limit=limit * 2, filter=filter)
        sparse_hits = self.query_sparse(
            collection, sparse, limit=limit * 2, filter=filter, query_text=query_text,
        )
        return _rrf_fuse(dense_hits, sparse_hits, limit=limit)

    def retrieve(self, collection: str, ids: list[str], *,
                 with_payload: bool = True, with_vectors: bool = False) -> list[PointStruct]:
        placeholders = ",".join("?" for _ in ids)
        cols = "id, payload" + (", embedding" if with_vectors else "")
        rows = self._conn.execute(
            f"SELECT {cols} FROM vec_{_s(collection)} WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        results = []
        for row in rows:
            payload = json.loads(row[1]) if with_payload and row[1] else None
            vec = None
            if with_vectors and len(row) > 2 and row[2]:
                # Parse "[0.1,0.2,...]" back to list[float]
                try:
                    vec = [float(x) for x in row[2].strip("[]").split(",")]
                except Exception:
                    pass
            results.append(PointStruct(id=str(row[0]), vector=vec or [], payload=payload))
        return results

    def scroll(self, collection: str, *, limit: int = 1000, offset=None,
               with_payload: bool = True,
               filter: Filter | None = None,
    ) -> tuple[list[PointStruct], Any]:
        """Paginated scan via rowid cursor.

        Returns ``next_offset`` = max rowid of this batch (or None when exhausted),
        so callers looping on offset retrieve every point. Returning None here
        (the old behaviour) silently truncated collections larger than ``limit``.
        """
        where, params = _filter_to_sql(filter, "payload")
        where_clause = f"WHERE {where}" if where else ""
        if offset is not None:
            where_clause += f" AND rowid > {int(offset)}" if where_clause else f"WHERE rowid > {int(offset)}"
        params.append(limit)
        try:
            rows = self._conn.execute(
                f"SELECT rowid, id, payload FROM vec_{_s(collection)} {where_clause} ORDER BY rowid LIMIT ?",
                tuple(params),
            ).fetchall()
        except Exception as e:
            logger.error(f"scroll failed: {e}")
            return [], None
        results = []
        max_rowid = 0
        for row in rows:
            if row[0] > max_rowid:
                max_rowid = row[0]
            payload = json.loads(row[2]) if with_payload and row[2] else None
            results.append(PointStruct(id=str(row[1]), vector=[], payload=payload))
        # Signal more pages iff we filled the batch
        next_offset = max_rowid if (results and len(results) >= limit) else None
        return results, next_offset

    def get_collection(self, name: str) -> CollectionInfo:
        try:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM vec_{_s(name)}"
            ).fetchone()
            points_count = row[0] if row else 0
        except Exception:
            points_count = 0

        # Try to get dimension from a sample vector
        dim = None
        try:
            row = self._conn.execute(
                f"SELECT embedding FROM vec_{_s(name)} WHERE embedding IS NOT NULL LIMIT 1"
            ).fetchone()
            if row and row[0]:
                # Parse "[0.1, 0.2, ...]" → count commas + 1
                dim = row[0].strip("[]").count(",") + 1
        except Exception:
            pass

        has_sparse = False
        try:
            fts_check = self._conn.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='vec_{_s(name)}_fts'"
            ).fetchone()
            has_sparse = bool(fts_check)
        except Exception:
            pass

        return CollectionInfo(
            name=name,
            points_count=points_count,
            named_vectors={"": dim} if dim else {},
            has_sparse=has_sparse,
            distance="Cosine",
        )

    def list_collections(self) -> list[str]:
        """Return original collection names (reverse-mapped from sanitized table names)."""
        self._ensure_map_table()
        # Get sanitized names from sqlite_master, then look up originals
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vec_%' "
            "AND name NOT LIKE '%_fts%' AND name != '_vec_collection_map'"
        ).fetchall()
        results = []
        for r in rows:
            sanitized = r[0][4:]  # strip "vec_" prefix
            results.append(self._lookup_original_name(sanitized))
        return results

    # ------------------------------------------------------------------
    # Migration (R1: default with_vectors=False)
    # ------------------------------------------------------------------

    def iter_all(self, collection: str, *, with_vectors: bool = False) -> Iterator[PointStruct]:
        cols = "id, payload" + (", embedding" if with_vectors else "")
        cursor = self._conn.execute(f"SELECT {cols} FROM vec_{_s(collection)}")
        while True:
            batch = cursor.fetchmany(100)
            if not batch:
                break
            for row in batch:
                payload = json.loads(row[1]) if row[1] else {}
                vec: list[float] = []
                if with_vectors and len(row) > 2 and row[2]:
                    # Parse "[0.1,0.2,...]" → list[float]
                    try:
                        vec = [float(x) for x in row[2].strip("[]").split(",")]
                    except Exception:
                        pass
                yield PointStruct(id=str(row[0]), vector=vec, payload=payload)

    def bulk_upsert(self, collection: str, points: Iterable[PointStruct]) -> int:
        count = 0
        batch = []
        for p in points:
            batch.append(p)
            if len(batch) >= 100:
                self.upsert(collection, batch)
                count += len(batch)
                batch = []
        if batch:
            self.upsert(collection, batch)
            count += len(batch)
        return count


# ---------------------------------------------------------------------------
# Filter translation helper
# ---------------------------------------------------------------------------

def _filter_to_sql(f: Filter | None, payload_col: str = "payload") -> tuple[str, list]:
    """Translate neutral Filter → SQL WHERE clause + params.

    Uses json_each for tag array containment (MatchContains).
    Uses json_extract for scalar field access.
    """
    if f is None or (not f.must and not f.should and not f.must_not):
        return "", []

    parts = []
    params: list = []

    for c in f.must:
        cond, cparams = _condition_to_sql(c, payload_col, negated=False)
        if cond:
            parts.append(cond)
            params.extend(cparams)

    for c in f.must_not:
        cond, cparams = _condition_to_sql(c, payload_col, negated=True)
        if cond:
            parts.append(f"NOT ({cond})")
            params.extend(cparams)

    return " AND ".join(parts), params


def _condition_to_sql(c: FieldCondition, payload_col: str, negated: bool) -> tuple[str, list]:
    """Translate one FieldCondition to SQL.

    Note: json path keys ($.{key}) are embedded directly in SQL — keys come from
    the application's payload schema, not from untrusted user input. The match
    VALUES are still parameterized.
    """
    key = c.key
    # Sanitize key for SQL embedding (alphanumeric + underscore only)
    if not key.replace("_", "").isalnum():
        return "", []  # skip unsafe keys

    if c.range_:
        parts = []
        params = []
        for val, op in [
            (c.range_.gt, ">"),
            (c.range_.gte, ">="),
            (c.range_.lt, "<"),
            (c.range_.lte, "<="),
        ]:
            if val is not None:
                parts.append(f"CAST(json_extract({payload_col}, '$.{key}') AS REAL) {op} ?")
                params.append(val)
        return " AND ".join(parts), params

    if c.match is None:
        return "", []

    if isinstance(c.match, MatchValue):
        return rf"CAST(json_extract({payload_col}, '$.{key}') AS TEXT) = ?", [str(c.match.value)]
    elif isinstance(c.match, MatchText):
        return rf"LOWER(CAST(json_extract({payload_col}, '$.{key}') AS TEXT)) LIKE LOWER(?)", [
            f"%{c.match.text}%"
        ]
    elif isinstance(c.match, MatchAny):
        placeholders = ",".join("?" for _ in c.match.values)
        return rf"CAST(json_extract({payload_col}, '$.{key}') AS TEXT) IN ({placeholders})", [
            *[str(v) for v in c.match.values]
        ]
    elif isinstance(c.match, MatchContains):
        return (rf"EXISTS (SELECT 1 FROM json_each(json_extract({payload_col}, '$.{key}')) WHERE value = ?)",
                [c.match.value])
    return "", []
