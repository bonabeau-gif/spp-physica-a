"""Content-addressed result store, held in the main process.

Workers are stateless and return raw results; the main process is the only writer,
so caching is simply a dict keyed by (u_hash, protocol_hash, seed) persisted to a
pickle shard. Continuation and localization revisit nearby points constantly, so
cross-call hits (not within-call) are where the savings are. Hit rate is logged by
callers. Resumability: load() on start, flush() after each accepted boundary point.
"""
from __future__ import annotations
import pickle
import gzip
from pathlib import Path
import threading


class RunCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.store: dict = {}
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        if self.path.exists() or Path(str(self.path) + ".gz").exists():
            self.load()

    def load(self):
        source = self.path if self.path.exists() else Path(str(self.path) + ".gz")
        opener = gzip.open if source.suffix == ".gz" else open
        with opener(source, "rb") as fh:
            self.store = pickle.load(fh)

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(self.store, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.path)

    def get(self, key):
        v = self.store.get(key)
        with self._lock:
            if v is None:
                self.misses += 1
            else:
                self.hits += 1
        return v

    def put(self, key, value):
        self.store[key] = value

    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0
