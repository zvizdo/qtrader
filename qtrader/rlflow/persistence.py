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
