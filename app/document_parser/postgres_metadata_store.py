"""PostgreSQL metadata store helpers."""

from typing import Sequence

from .types import StoredNode


def _json_payload(value):
    from psycopg.types.json import Json

    return Json(value)


class PostgresMetadataStore:
    def __init__(self, *, pg_dsn: str, table_prefix: str) -> None:
        import psycopg

        self._pg_conn = psycopg.connect(pg_dsn)
        self._pg_conn.autocommit = True
        self._table = f"{table_prefix}_nodes"
        self._ensure_tables()

    def close(self) -> None:
        try:
            self._pg_conn.close()
        except Exception:
            pass

    def _ensure_tables(self) -> None:
        with self._pg_conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    node_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    child_ids JSONB NOT NULL
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_parent_idx ON {self._table}(parent_id)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_kind_idx ON {self._table}(kind)"
            )

    def upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:
        with self._pg_conn.cursor() as cur:
            for node in nodes:
                cur.execute(
                    f"""
                    INSERT INTO {self._table} (
                        node_id, parent_id, kind, title, text, metadata, child_ids
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (node_id) DO UPDATE SET
                        parent_id = EXCLUDED.parent_id,
                        kind = EXCLUDED.kind,
                        title = EXCLUDED.title,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        child_ids = EXCLUDED.child_ids
                    """,
                    (
                        node.node_id,
                        node.parent_id,
                        node.kind.value,
                        node.title,
                        node.text,
                        _json_payload(node.metadata),
                        _json_payload(list(node.child_ids)),
                    ),
                )

    def fetch_rows(self, node_ids: Sequence[str]) -> dict[str, dict]:
        if not node_ids:
            return {}

        rows: dict[str, dict] = {}
        with self._pg_conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT node_id, parent_id, kind, title, text, metadata, child_ids
                FROM {self._table}
                WHERE node_id = ANY(%s)
                """,
                (list(node_ids),),
            )
            for row in cur.fetchall():
                rows[row[0]] = {
                    "node_id": row[0],
                    "parent_id": row[1],
                    "kind": row[2],
                    "title": row[3],
                    "text": row[4],
                    "metadata": row[5] or {},
                    "child_ids": row[6] or [],
                }
        return rows
