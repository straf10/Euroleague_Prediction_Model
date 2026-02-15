from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import pandas as pd
import json


class Cache:
    """Simple filesystem cache.

    Stores DataFrames as pickle for exact dtypes and fast IO.
    Stores misc objects as JSON where possible.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _df_path(self, key: str) -> Path:
        return self.root / f"{key}.pkl"

    def has_df(self, key: str) -> bool:
        return self._df_path(key).exists()

    def load_df(self, key: str) -> pd.DataFrame:
        p = self._df_path(key)
        if not p.exists():
            raise FileNotFoundError(f"Cache miss: {p}")
        return pd.read_pickle(p)

    def save_df(self, key: str, df: pd.DataFrame) -> Path:
        p = self._df_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(p)
        return p

    def _json_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def has_json(self, key: str) -> bool:
        return self._json_path(key).exists()

    def load_json(self, key: str) -> Any:
        p = self._json_path(key)
        if not p.exists():
            raise FileNotFoundError(f"Cache miss: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    def save_json(self, key: str, obj: Any) -> Path:
        p = self._json_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def path(self, key: str) -> Path:
        # returns best guess (df path)
        return self._df_path(key)
