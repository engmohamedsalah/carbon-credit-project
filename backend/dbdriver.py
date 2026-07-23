"""Database connection factory: Turso (libSQL) in production, sqlite3 locally.

The libSQL client returns plain tuples and ignores ``row_factory``
(tursodatabase/libsql-python#24), while this app reads rows by name everywhere
(``dict(row)``, ``row["col"]``, ``row[0]``). So in Turso mode we wrap the
connection/cursor to rebuild name-accessible rows from ``cursor.description``,
keeping every existing call site unchanged.

If ``TURSO_DATABASE_URL`` is unset (local dev), we return a real sqlite3
connection with ``row_factory = sqlite3.Row`` — i.e. today's behaviour exactly.
Only ``?`` positional placeholders are used by the app (named params are not
supported by libSQL), verified before migration.
"""
import os
import sqlite3

TURSO_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")


class _Row(dict):
    """dict row that also supports positional access (row[0]) for legacy sites."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class _Cursor:
    def __init__(self, cur):
        self._cur = cur

    def _wrap(self, row):
        if row is None:
            return None
        cols = [d[0] for d in self._cur.description]
        return _Row(zip(cols, row))

    def fetchone(self):
        return self._wrap(self._cur.fetchone())

    def fetchall(self):
        return [self._wrap(r) for r in self._cur.fetchall()]

    def __iter__(self):
        return (self._wrap(r) for r in self._cur)

    def __getattr__(self, name):  # lastrowid, rowcount, description, ...
        return getattr(self._cur, name)


class _Conn:
    def __init__(self, raw):
        self._raw = raw

    def execute(self, sql, params=()):
        return _Cursor(self._raw.execute(sql, params))

    def __getattr__(self, name):  # commit, rollback, close, ...
        return getattr(self._raw, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._raw.commit()
        else:
            self._raw.rollback()
        return False


def connect(db_path: str):
    """Return a connection that yields name-accessible rows in both modes.

    ``db_path`` is used only for the local sqlite3 fallback; it is ignored when
    connecting to Turso.
    """
    if TURSO_URL:
        import libsql  # lazy import so local dev needn't install it

        return _Conn(libsql.connect(database=TURSO_URL, auth_token=TURSO_TOKEN))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
