"""
Concrete backend implementations.

This package is NOT scanned by the tool discovery system.
Callers should never import from here directly -- use the factories in
sql_store.py / vector_store.py / store_factory.py instead.

Phase 0 of the backend abstraction plan -- empty package marker.
Concrete impls land in later phases:

  Phase 1: postgres_sql.py     (PostgresSqlStore)
  Phase 2: qdrant_vector.py    (QdrantVectorStore)
  Phase 3: turso_sql.py        (TursoSqlStore)
           turso_vector.py     (TursoVectorStore)
           postgres_vector.py  (PostgresVectorStore)
"""
