"""A column store: one folder, a manifest, and shards of .npy columns (memory-mapped on read).

    st = MomentStore("interp/moments")              # a folder (made on the first write)
    st.write_shard({"obs": obs, "game": game, ...}, meta={"opponent": "self"})
    st.columns()                                    # name -> (dtype, per-row shape)
    cols = st.load(["obs", "game"], rows=idx)       # rows across all shards (an index array, or None for all)
    for start, part in st.iter(["obs"], batch=65536): ...   # streaming, in order

Every shard holds the same columns. Nothing here knows about any game or bot: :class:`cds_interpret.Moments`
is built on it.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .errors import CDSInterpretError

MANIFEST = "manifest.json"
FORMAT = "cds-interpret.store"


class MomentStore:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self._m: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"MomentStore({str(self.root)!r}: {len(self)} rows in {len(self.shards)} shards)"

    # ---- manifest
    @property
    def manifest(self) -> dict[str, Any]:
        if self._m is None:
            p = self.root / MANIFEST
            if p.exists():
                self._m = json.loads(p.read_text(encoding="utf-8"))
            else:
                self._m = {"format": FORMAT, "version": 1, "columns": {}, "shards": [], "meta": {}}
        return self._m

    def exists(self) -> bool:
        return (self.root / MANIFEST).exists()

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=1), encoding="utf-8")
        os.replace(tmp, self.root / MANIFEST)

    def set_meta(self, **meta: Any) -> None:
        self.manifest["meta"].update(meta)
        self._save()

    @property
    def meta(self) -> dict[str, Any]:
        return self.manifest["meta"]

    def columns(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        return {k: (v["dtype"], tuple(v["shape"])) for k, v in self.manifest["columns"].items()}

    def __len__(self) -> int:
        return sum(s["rows"] for s in self.manifest["shards"])

    @property
    def shards(self) -> list[dict[str, Any]]:
        return self.manifest["shards"]

    # ---- writing
    def write_shard(self, cols: dict[str, np.ndarray], meta: dict[str, Any] | None = None, name: str | None = None) -> str:
        if not cols:
            raise ValueError("write_shard got no columns")
        lengths = {k: len(v) for k, v in cols.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"columns of different lengths: {lengths}")
        rows = next(iter(lengths.values()))
        known = self.manifest["columns"]
        if known and set(known) != set(cols):
            raise CDSInterpretError(f"this store's shards have the columns {sorted(known)}; the new shard has "
                                    f"{sorted(cols)}. Write it to a new folder, or give it the same columns.")
        specs = {}
        for k, v in cols.items():
            arr = np.asarray(v)
            spec = {"dtype": str(arr.dtype), "shape": list(arr.shape[1:])}
            if k in known and known[k] != spec:
                raise CDSInterpretError(f"column {k}: {spec} differs from the store's {known[k]}")
            specs[k] = spec
        name = name or f"shard_{len(self.manifest['shards']):05d}"
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        for k, v in cols.items():
            np.save(d / f"{k}.npy", np.ascontiguousarray(v))
        known.update(specs)
        self.manifest["shards"].append({"name": name, "rows": int(rows), "meta": meta or {}})
        self._save()
        return name

    # ---- reading
    def _col(self, shard: dict[str, Any], col: str) -> np.ndarray:
        return np.load(self.root / shard["name"] / f"{col}.npy", mmap_mode="r")

    def load(self, cols: list[str], rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Columns as arrays in memory: all rows, or the global row indices ``rows`` (any order; returned in that
        order)."""
        missing = [c for c in cols if c not in self.manifest["columns"]]
        if missing:
            raise KeyError(f"no columns {missing} in {self.root} (it has {sorted(self.manifest['columns'])})")
        shards = self.manifest["shards"]
        starts = np.cumsum([0] + [s["rows"] for s in shards])
        if rows is None:
            if not shards:
                return {c: np.zeros((0, *self.manifest["columns"][c]["shape"]), self.manifest["columns"][c]["dtype"])
                        for c in cols}
            return {c: np.concatenate([np.asarray(self._col(s, c)) for s in shards]) for c in cols}
        rows = np.asarray(rows, dtype=np.int64)
        which = np.searchsorted(starts, rows, side="right") - 1
        out = {}
        for c in cols:
            dt, shp = self.manifest["columns"][c]["dtype"], tuple(self.manifest["columns"][c]["shape"])
            arr = np.empty((len(rows), *shp), dtype=dt)
            for si in np.unique(which):
                m = which == si
                arr[m] = self._col(shards[si], c)[rows[m] - starts[si]]
            out[c] = arr
        return out

    def iter(self, cols: list[str], batch: int = 65536) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """(global start row, {col: array}) in order, at most ``batch`` rows at a time."""
        start = 0
        for s in self.manifest["shards"]:
            mm = {c: self._col(s, c) for c in cols}
            for i in range(0, s["rows"], batch):
                yield start + i, {c: np.asarray(m[i:i + batch]) for c, m in mm.items()}
            start += s["rows"]

    def shard_meta(self) -> list[dict[str, Any]]:
        return [s.get("meta", {}) for s in self.manifest["shards"]]

    def adopt(self, other: MomentStore, move: bool = False) -> list[str]:
        """Take in the shards of another store (recorded elsewhere, say) that this one lacks - shards with the same
        meta are taken to be the same. Their folders are copied (or moved) under new names and appended; the
        stores' columns must match. Returns the new shard names."""
        have = [json.dumps(s.get("meta", {}), sort_keys=True) for s in self.manifest["shards"]]
        known = self.manifest["columns"]
        if known and known != other.manifest["columns"]:
            raise CDSInterpretError("the two stores have different columns")
        new = []
        for s in other.manifest["shards"]:
            if json.dumps(s.get("meta", {}), sort_keys=True) in have or not (other.root / s["name"]).exists():
                continue
            name = f"shard_{len(self.manifest['shards']):05d}"
            self.root.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(other.root / s["name"]), str(self.root / name))
            else:
                shutil.copytree(other.root / s["name"], self.root / name)
            self.manifest["shards"].append({"name": name, "rows": s["rows"], "meta": s.get("meta", {})})
            new.append(name)
        if not known:
            self.manifest["columns"] = dict(other.manifest["columns"])
        for k, v in other.manifest.get("meta", {}).items():
            self.manifest["meta"].setdefault(k, v)
        self._save()
        return new
