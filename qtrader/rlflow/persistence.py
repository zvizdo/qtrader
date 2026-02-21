import gzip
from io import BytesIO
import json
from json import JSONEncoder
import os
from pathlib import Path
import pickle
import sqlite3
from datetime import datetime
from typing import List, Optional, Any
from numpy import ndarray, int64, float32
import msgpack


class BasePersistenceProvider(object):

    def list(self, prefix: str) -> List[dict]:
        raise NotImplementedError()

    def persist_dict(self, name: str, obj: dict) -> None:
        raise NotImplementedError()

    def load_dict(self, name: str) -> Optional[dict]:
        raise NotImplementedError()

    def persist_obj(self, name: str, obj: Any) -> None:
        raise NotImplementedError()

    def load_obj(self, name: str) -> Any:
        raise NotImplementedError()

    def delete(self, name: str) -> None:
        raise NotImplementedError()

    def root_join(self, name: str) -> str:
        raise NotImplementedError()


class PersistenceJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()

        elif isinstance(o, ndarray):
            return o.tolist()

        elif isinstance(o, int64):
            return int(o)

        elif isinstance(o, float32):
            return float32(o)

        return JSONEncoder.default(self, o)


class NoPersistenceProvider(BasePersistenceProvider):

    def list(self, prefix: str) -> List[dict]:
        pass

    def persist_dict(self, name: str, obj: dict) -> None:
        pass

    def load_dict(self, name: str) -> Optional[dict]:
        return None

    def persist_obj(self, name: str, obj: Any) -> None:
        pass

    def load_obj(self, name: str) -> Any:
        return None

    def delete(self, name: str) -> None:
        pass

    def root_join(self, name: str) -> str:
        return None


class FileSystemPersistenceProvider(BasePersistenceProvider):

    def __init__(self, root: str):
        super(FileSystemPersistenceProvider, self).__init__()
        self.root = root

    def list(self, prefix: str) -> List[dict]:
        for f in os.listdir(self.root):
            if f.startswith(prefix):
                yield f.split(".")[0]

    def persist_dict(self, name: str, obj: dict) -> None:
        path = os.path.join(self.root, f"{name}.gz")
        with gzip.open(path, "wb") as gz:
            gz.write(json.dumps(obj, cls=PersistenceJSONEncoder).encode("utf-8"))

    def load_dict(self, name: str) -> dict:
        path = os.path.join(self.root, f"{name}.gz")
        with gzip.open(path, "rb") as gz:
            return json.load(gz)

    def persist_obj(self, name: str, obj: Any) -> None:
        path = os.path.join(self.root, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name: str) -> Any:
        path = os.path.join(self.root, f"{name}.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)

    def delete(self, name: str) -> None:
        try:
            path = os.path.join(self.root, f"{name}.gz")
            os.remove(path)
            return
        except:
            pass

        try:
            path = os.path.join(self.root, f"{name}.pkl")
            os.remove(path)
        except:
            pass

    def root_join(self, name: str) -> str:
        return os.path.join(self.root, name)


class SQLitePersistenceProvider(BasePersistenceProvider):
    def __init__(self, root: str, dbname: str = "db.sqlite"):
        super(SQLitePersistenceProvider, self).__init__()
        self.root = root
        self.dbname = dbname
        self.dbc = sqlite3.connect(self.root_join(dbname))
        self.dbexe = self.dbc.cursor()

        # WAL mode: readers don't block writers, NORMAL sync reduces fsyncs
        self.dbexe.execute("PRAGMA journal_mode=WAL;")
        self.dbexe.execute("PRAGMA synchronous=NORMAL;")

        self.dbexe.execute(
            """
            CREATE TABLE IF NOT EXISTS data (
                id TEXT PRIMARY KEY,
                payload BLOB
            );
            """
        )

    def list(self, prefix: str) -> List[dict]:
        id = {"prefix": f"{prefix}%"}
        sql = f"SELECT id FROM data WHERE id LIKE :prefix ORDER BY id DESC"
        self.dbexe.execute(sql, id)
        for r in self.dbexe:
            yield r[0]

    def persist_dict(self, name: str, obj: dict) -> None:
        id = f"{name}"
        with BytesIO() as pl:
            with gzip.open(pl, "wb") as gz:
                gz.write(json.dumps(obj, cls=PersistenceJSONEncoder).encode("utf-8"))

            pl.seek(0)
            self.dbexe.execute(
                """INSERT INTO data (id, payload) VALUES (?, ?)
                   ON CONFLICT(id) DO UPDATE SET payload=excluded.payload; 
                """,
                (id, sqlite3.Binary(pl.read())),
            )
            self.dbc.commit()

    def load_dict(self, name: str) -> dict:
        id = {"id": f"{name}"}
        sql = f"SELECT payload FROM data WHERE id = :id"
        self.dbexe.execute(sql, id)
        pl = self.dbexe.fetchone()[0]

        with BytesIO(pl) as s:
            with gzip.open(s, "rb") as gz:
                return json.load(gz)

    def persist_obj(self, name: str, obj: Any) -> None:
        id = f"{name}"
        self.dbexe.execute(
            """INSERT INTO data (id, payload) VALUES (?, ?)
               ON CONFLICT(id) DO UPDATE SET payload=excluded.payload; 
            """,
            (id, sqlite3.Binary(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))),
        )
        self.dbc.commit()

    def load_obj(self, name: str) -> Any:
        id = {"id": f"{name}"}
        sql = f"SELECT payload FROM data WHERE id = :id"
        self.dbexe.execute(sql, id)
        pl = self.dbexe.fetchone()[0]

        with BytesIO(pl) as s:
            return pickle.load(s)

    def delete(self, name: str) -> None:
        id = {"id": f"{name}"}
        sql = f"DELETE FROM data WHERE id = :id"
        self.dbexe.execute(sql, id)
        self.dbc.commit()

    def root_join(self, name: str) -> str:
        return os.path.join(self.root, name)


class CachedSQLitePersistenceProvider(SQLitePersistenceProvider):
    """SQLite provider with in-memory read cache, batched writes, and msgpack serialization."""

    def __init__(self, root: str, dbname: str = "db.sqlite", cache_size: int = 512, flush_interval: int = 50):
        super().__init__(root, dbname)
        self._write_buffer = {}           # name -> serialized payload
        self._read_cache = {}             # name -> deserialized dict
        self._cache_size = cache_size
        self._flush_interval = flush_interval
        self._write_count = 0

    def persist_dict(self, name: str, obj: dict) -> None:
        payload = msgpack.packb(obj, use_bin_type=True)
        self._write_buffer[name] = payload
        self._read_cache[name] = obj      # update read cache immediately
        self._write_count += 1
        if self._write_count >= self._flush_interval:
            self.flush()

    def load_dict(self, name: str) -> dict:
        # 1. Check in-memory cache
        if name in self._read_cache:
            return self._read_cache[name]

        # 2. Check write buffer (written but not yet flushed)
        if name in self._write_buffer:
            obj = msgpack.unpackb(self._write_buffer[name], raw=False)
            self._read_cache[name] = obj
            return obj

        # 3. Fall through to SQLite
        row = self.dbexe.execute(
            "SELECT payload FROM data WHERE id = ?", (name,)
        ).fetchone()
        if row is None:
            raise KeyError(name)

        # Try msgpack first, fall back to legacy gzip+json for existing data
        try:
            obj = msgpack.unpackb(row[0], raw=False)
        except (msgpack.exceptions.UnpackValueError, msgpack.exceptions.ExtraData):
            with BytesIO(row[0]) as s:
                with gzip.open(s, "rb") as gz:
                    obj = json.load(gz)

        self._read_cache[name] = obj
        self._evict_if_needed()
        return obj

    def flush(self):
        """Flush all buffered writes to SQLite in a single transaction."""
        if not self._write_buffer:
            return
        self.dbexe.executemany(
            """INSERT INTO data (id, payload) VALUES (?, ?)
               ON CONFLICT(id) DO UPDATE SET payload=excluded.payload;""",
            [(k, sqlite3.Binary(v)) for k, v in self._write_buffer.items()],
        )
        self.dbc.commit()  # single fsync for the entire batch
        self._write_buffer.clear()
        self._write_count = 0

    def _evict_if_needed(self):
        """Evict oldest entries from read cache if over capacity."""
        while len(self._read_cache) > self._cache_size:
            self._read_cache.pop(next(iter(self._read_cache)))


class LeanSQLitePersistenceProvider(SQLitePersistenceProvider):

    def __init__(self, prefix: str, lean_obj_store, dbname="db.sqlite"):
        self.lean_obj_store = lean_obj_store
        self.prefix = prefix

        db_path = Path(self._request_path(dbname))
        super().__init__(db_path.parent, db_path.name)

    def _request_path(self, filename):
        return self.lean_obj_store.get_file_path(f"{self.prefix}/{filename}")

    def root_join(self, name):
        return self._request_path(name)


class LeanCachedSQLitePersistenceProvider(CachedSQLitePersistenceProvider):
    """Lean-integrated version of CachedSQLitePersistenceProvider."""

    def __init__(self, prefix: str, lean_obj_store, dbname="db.sqlite", cache_size=512, flush_interval=50):
        self.lean_obj_store = lean_obj_store
        self.prefix = prefix

        db_path = Path(self._request_path(dbname))
        super().__init__(db_path.parent, db_path.name, cache_size=cache_size, flush_interval=flush_interval)

    def _request_path(self, filename):
        return self.lean_obj_store.get_file_path(f"{self.prefix}/{filename}")

    def root_join(self, name):
        return self._request_path(name)
