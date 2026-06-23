"""
PostgresVectorStore — VectorStore backed by PostgreSQL + pgvector extension.

Phase 3 of the backend abstraction plan.

Requires:
- PostgreSQL 12+ (for tsvector GENERATED ... STORED)
- pgvector extension installed on the server
"""
from __future__ import annotations

import json
import logging
from typing import Iterator, Iterable, Any

from shared.store_models import (
    PointStruct, ScoredPoint, Filter, FieldCondition,
    MatchValue, MatchText, MatchAny, MatchContains, Range,
    CollectionInfo, SparseVector,
)

logger = logging.getLogger(__name__)


class PostgresVectorStore:
    """VectorStore backed by PostgreSQL + pgvector."""

    def __init__(self, dsn: str):
        from psycopg_pool import ConnectionPool
        from psycopg.rows import dict_row
        self._pool = ConnectionPool(
            conninfo=dsn, min_size=1, max_size=5,
            kwargs={"row_factory": dict_row}, open=False,
        )
        self._pool.open(wait=True)
        # Ensure pgvector extension
        with self._pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # R6: PG 12+ check
            ver = conn.execute("SELECT current_setting('server_version_num')::int AS v").fetchone()
            if ver and ver["v"] < 120000:
                raise RuntimeError(f"PostgresVectorStore requires PG 12+ (have {ver['v']})")

    def _schema_sql(self, name: str, dim: int) -> str:
        return f"""
            CREATE TABLE IF NOT EXISTS vec_{name} (
                id          TEXT PRIMARY KEY,
                embedding   vector({dim}),
                payload     JSONB,
                search_vector tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(payload->>'text', ''))
                ) STORED,
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_{name}_embedding
                ON vec_{name} USING hnsw (embedding vector_cosine_ops);
            CREATE INDEX IF NOT EXISTS idx_{name}_search
                ON vec_{name} USING gin (search_vector);
        """

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def ensure_collection(
        self, name: str, *,
        dense_dim: int | None = None,
        sparse: bool = False,
        distance: str = "Cosine",
    ) -> None:
        if dense_dim is None:
            # Sparse-only: use tsvector for full-text, no dense vector
            with self._pool.connection() as conn:
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS vec_{name} (
                        id TEXT PRIMARY KEY,
                        payload JSONB,
                        search_vector tsvector GENERATED ALWAYS AS (
                            to_tsvector('english', coalesce(payload->>'text', ''))
                        ) STORED
                    );
                    CREATE INDEX IF NOT EXISTS idx_{name}_search
                        ON vec_{name} USING gin (search_vector);
                """)
            return

        # Dense or hybrid
        with self._pool.connection() as conn:
            conn.execute(self._schema_sql(name, dense_dim))

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert(self, collection: str, points: list[PointStruct]) -> None:
        with self._pool.connection() as conn:
            for p in points:
                vec_str = _serialize_vector(p.vector)
                payload_json = json.dumps(p.payload or {})
                conn.execute(f"""
                    INSERT INTO vec_{collection} (id, embedding, payload)
                    VALUES (%s, %s::vector, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        payload = EXCLUDED.payload
                """, (str(p.id), vec_str, payload_json))

    def set_payload(self, collection: str, payload: dict, *, ids: list[str]) -> None:
        payload_json = json.dumps(payload)
        with self._pool.connection() as conn:
            conn.execute(
                f"UPDATE vec_{collection} SET payload = %s::jsonb WHERE id = ANY(%s)",
                (payload_json, ids),
            )

    def delete(self, collection: str, *,
               ids: list[str] | None = None,
               filter: Filter | None = None) -> None:
        with self._pool.connection() as conn:
            if ids:
                conn.execute(
                    f"DELETE FROM vec_{collection} WHERE id = ANY(%s)",
                    (ids,),
                )
            elif filter:
                where, params = _filter_to_sql(filter)
                if where:
                    conn.execute(f"DELETE FROM vec_{collection} WHERE {where}", tuple(params))

    def delete_collection(self, name: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS vec_{name}")

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def query_dense(self, collection: str, vec: list[float], *,
                    limit: int = 10, filter: Filter | None = None,
                    using: str | None = None) -> list[ScoredPoint]:
        vec_str = _serialize_vector(vec)
        where, params = _filter_to_sql(filter)
        where_clause = f"AND {where}" if where else ""
        params = [vec_str, vec_str, limit] + params
        with self._pool.connection() as conn:
            rows = conn.execute(f"""
                SELECT id, payload,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM vec_{collection}
                WHERE 1 - (embedding <=> %s::vector) > -1
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, tuple(params)).fetchall()
        results = []
        for row in rows:
            results.append(ScoredPoint(
                id=str(row["id"]),
                score=float(row["similarity"]),
                payload=row["payload"] or {},
            ))
        return results

    def query_sparse(self, collection: str, sparse: SparseVector, *,
                     limit: int = 10, filter: Filter | None = None) -> list[ScoredPoint]:
        # Convert sparse vector indices to a tsquery
        # Without a vocabulary, we can't reconstruct terms. Use a basic query.
        terms = " | ".join(f"x{i}" for i in sparse.indices[:10])
        where, params = _filter_to_sql(filter)
        where_clause = f"AND {where}" if where else ""
        params = [terms, terms, limit] + params
        with self._pool.connection() as conn:
            rows = conn.execute(f"""
                SELECT id, payload, ts_rank(search_vector, to_tsquery(%s)) AS score
                FROM vec_{collection}
                WHERE search_vector @@ to_tsquery(%s)
                {where_clause}
                ORDER BY score DESC
                LIMIT %s
            """, tuple(params)).fetchall()
        return [ScoredPoint(
            id=str(row["id"]),
            score=float(row["score"]),
            payload=row["payload"] or {},
        ) for row in rows]

    def query_hybrid(self, collection: str, dense: list[float],
                     sparse: SparseVector, *, limit: int = 10,
                     filter: Filter | None = None) -> list[ScoredPoint]:
        vec_str = _serialize_vector(dense)
        terms = " | ".join(f"x{i}" for i in sparse.indices[:10])
        where, params = _filter_to_sql(filter)
        where_clause = f"AND {where}" if where else ""
        params = [vec_str, vec_str, terms, terms, limit] + params
        with self._pool.connection() as conn:
            rows = conn.execute(f"""
                SELECT id, payload,
                    (1 - (embedding <=> %s::vector)) * 0.5 +
                    ts_rank(search_vector, to_tsquery(%s)) * 0.5 AS combined
                FROM vec_{collection}
                WHERE search_vector @@ to_tsquery(%s)
                {where_clause}
                ORDER BY combined DESC
                LIMIT %s
            """, tuple(params)).fetchall()
        return [ScoredPoint(
            id=str(row["id"]),
            score=float(row["combined"]),
            payload=row["payload"] or {},
        ) for row in rows]

    def retrieve(self, collection: str, ids: list[str], *,
                 with_payload: bool = True, with_vectors: bool = False) -> list[PointStruct]:
        with self._pool.connection() as conn:
            cols = "id, payload" + (", embedding" if with_vectors else "")
            rows = conn.execute(
                f"SELECT {cols} FROM vec_{collection} WHERE id = ANY(%s)",
                (ids,),
            ).fetchall()
        results = []
        for row in rows:
            vec = None
            if with_vectors and row.get("embedding") is not None:
                try:
                    vec = [float(x) for x in str(row["embedding"]).strip("[]").split(",")]
                except Exception:
                    pass
            results.append(PointStruct(
                id=str(row["id"]),
                vector=vec or [],
                payload=row["payload"] if with_payload else None,
            ))
        return results

    def scroll(self, collection: str, *, limit: int = 1000, offset=None,
               with_payload: bool = True,
               filter: Filter | None = None,
    ) -> tuple[list[PointStruct], Any]:
        where, params = _filter_to_sql(filter)
        where_clause = f"WHERE {where}" if where else ""
        offset_clause = f"OFFSET {int(offset)}" if offset is not None else ""
        params = params + [limit]
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT id, payload FROM vec_{collection} {where_clause} "
                f"ORDER BY id LIMIT {int(limit)} {offset_clause}",
                tuple(params),
            ).fetchall()
        results = [
            PointStruct(id=str(r["id"]), vector=[], payload=r["payload"] if with_payload else None)
            for r in rows
        ]
        next_offset = (int(offset) + len(results)) if results else None
        return results, next_offset

    def get_collection(self, name: str) -> CollectionInfo:
        with self._pool.connection() as conn:
            count_row = conn.execute(f"SELECT COUNT(*) AS c FROM vec_{name}").fetchone()
            dim_row = conn.execute(
                f"SELECT vector_dims(embedding) AS d FROM vec_{name} WHERE embedding IS NOT NULL LIMIT 1"
            ).fetchone()
            sparse_row = conn.execute(
                f"SELECT 1 FROM information_schema.columns "
                f"WHERE table_name = 'vec_{name}' AND column_name = 'search_vector'"
            ).fetchone()
        return CollectionInfo(
            name=name,
            points_count=count_row["c"] if count_row else 0,
            named_vectors={"": dim_row["d"]} if dim_row and dim_row["d"] else {},
            has_sparse=bool(sparse_row),
            distance="Cosine",
        )

    def list_collections(self) -> list[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 'vec_%' AND table_schema = 'public'"
            ).fetchall()
        return [r["table_name"][4:] for r in rows]  # strip "vec_" prefix

    # ------------------------------------------------------------------
    # Migration (R1: default with_vectors=False)
    # ------------------------------------------------------------------

    def iter_all(self, collection: str, *, with_vectors: bool = False) -> Iterator[PointStruct]:
        cols = "id, payload" + (", embedding" if with_vectors else "")
        with self._pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT {cols} FROM vec_{collection} ORDER BY id"
            )
            while True:
                batch = cursor.fetchmany(100)
                if not batch:
                    break
                for row in batch:
                    vec = None
                    if with_vectors and row.get("embedding") is not None:
                        try:
                            vec = [float(x) for x in str(row["embedding"]).strip("[]").split(",")]
                        except Exception:
                            pass
                    yield PointStruct(
                        id=str(row["id"]),
                        vector=vec or [],
                        payload=row["payload"],
                    )

    def bulk_upsert(self, collection: str, points: Iterable[PointStruct]) -> int:
        batch = list(points)
        if batch:
            self.upsert(collection, batch)
        return len(batch)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_vector(vec: list[float] | dict[str, list[float]]) -> str:
    """Serialize a vector for pgvector's vector type (S2)."""
    if isinstance(vec, list):
        return "[" + ",".join(str(x) for x in vec) + "]"
    # Named dense — extract first list
    for v in vec.values():
        if isinstance(v, list):
            return "[" + ",".join(str(x) for x in v) + "]"
    return "[]"


def _filter_to_sql(f: Filter | None) -> tuple[str, list]:
    """Translate neutral Filter → SQL WHERE clause + params."""
    if f is None or (not f.must and not f.should and not f.must_not):
        return "", []
    parts = []
    params: list = []
    for c in f.must:
        cond, cparams = _condition_to_sql(c)
        if cond:
            parts.append(cond)
            params.extend(cparams)
    for c in f.must_not:
        cond, cparams = _condition_to_sql(c)
        if cond:
            parts.append(f"NOT ({cond})")
            params.extend(cparams)
    return " AND ".join(parts), params


def _condition_to_sql(c: FieldCondition) -> tuple[str, list]:
    """Translate one FieldCondition to PG SQL."""
    key = c.key
    if c.range_:
        parts = []
        params = []
        for op, val, pg_op in [
            ("gt", c.range_.gt, ">"),
            ("gte", c.range_.gte, ">="),
            ("lt", c.range_.lt, "<"),
            ("lte", c.range_.lte, "<="),
        ]:
            if val is not None:
                parts.append(f"(payload->>%s)::float {pg_op} %s")
                params.extend([key, val])
        return " AND ".join(parts), params
    if c.match is None:
        return "", []
    if isinstance(c.match, MatchValue):
        return "payload->>%s = %s", [key, str(c.match.value)]
    if isinstance(c.match, MatchText):
        return "payload->>%s ILIKE %s", [key, f"%{c.match.text}%"]
    if isinstance(c.match, MatchAny):
        placeholders = ",".join(["%s"] * len(c.match.values))
        return f"payload->>%s IN ({placeholders})", [key, *[str(v) for v in c.match.values]]
    if isinstance(c.match, MatchContains):
        return "payload->%s ? %s", [key, c.match.value]
    return "", []
