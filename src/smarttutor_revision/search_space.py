from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd

CONFIG_COLUMNS = ["top_k", "chunk_tokens", "overlap", "rerank_depth", "context_tokens"]


@dataclass(frozen=True)
class SearchSpace:
    """Mixed discrete SmartTutor search space.

    Semantics:
    * r = 0: no reranker; the retriever directly returns k chunks.
    * r > 0: the retriever obtains r candidates, the reranker scores them,
      and the top k chunks are retained. Therefore r >= k.

    The fifth normalized coordinate is decoded conditionally after k so that
    every decoded tuple is operationally feasible.
    """

    top_k_values: tuple[int, ...] = tuple(range(2, 13))
    chunk_values: tuple[int, ...] = tuple(range(128, 1025, 64))
    overlap_values: tuple[float, ...] = tuple(round(0.05 + 0.01 * i, 2) for i in range(26))
    rerank_max: int = 20
    context_values: tuple[int, ...] = tuple(range(1024, 8193, 256))

    @property
    def dimensions(self) -> int:
        return 5

    def rerank_values_for(self, top_k: int) -> np.ndarray:
        return np.asarray((0, *range(int(top_k), self.rerank_max + 1)), dtype=int)

    @property
    def size(self) -> int:
        valid_kr = sum(len(self.rerank_values_for(k)) for k in self.top_k_values)
        return (
            valid_kr
            * len(self.chunk_values)
            * len(self.overlap_values)
            * len(self.context_values)
        )

    def is_feasible(self, row: np.ndarray | tuple) -> bool:
        a = np.asarray(row)
        k, r = int(a[0]), int(a[3])
        return r == 0 or r >= k

    def decode(self, normalized: np.ndarray) -> np.ndarray:
        x = np.asarray(normalized, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != 5:
            raise ValueError("Expected five normalized variables.")
        x = np.clip(x, 0.0, np.nextafter(1.0, 0.0))

        top_grid = np.asarray(self.top_k_values)
        chunk_grid = np.asarray(self.chunk_values)
        overlap_grid = np.asarray(self.overlap_values)
        context_grid = np.asarray(self.context_values)

        top_idx = np.floor(x[:, 0] * len(top_grid)).astype(int)
        top_k = top_grid[top_idx]
        chunk = chunk_grid[np.floor(x[:, 1] * len(chunk_grid)).astype(int)]
        overlap = overlap_grid[np.floor(x[:, 2] * len(overlap_grid)).astype(int)]
        context = context_grid[np.floor(x[:, 4] * len(context_grid)).astype(int)]

        rerank = np.empty(len(x), dtype=int)
        for i, k in enumerate(top_k):
            grid = self.rerank_values_for(int(k))
            idx = int(np.floor(x[i, 3] * len(grid)))
            rerank[i] = int(grid[idx])

        return np.column_stack([top_k, chunk, overlap, rerank, context])

    def encode_midpoint(self, configurations: np.ndarray) -> np.ndarray:
        cfg = np.asarray(configurations)
        if cfg.ndim == 1:
            cfg = cfg.reshape(1, -1)
        cols = [[], [], [], [], []]
        top_lookup = {float(v): i for i, v in enumerate(self.top_k_values)}
        chunk_lookup = {float(v): i for i, v in enumerate(self.chunk_values)}
        overlap_lookup = {float(v): i for i, v in enumerate(self.overlap_values)}
        context_lookup = {float(v): i for i, v in enumerate(self.context_values)}
        for row in cfg:
            k, c, o, r, b = int(row[0]), int(row[1]), float(row[2]), int(row[3]), int(row[4])
            if not self.is_feasible(row):
                raise ValueError(f"Infeasible configuration: k={k}, r={r}; require r=0 or r>=k")
            cols[0].append((top_lookup[float(k)] + 0.5) / len(self.top_k_values))
            cols[1].append((chunk_lookup[float(c)] + 0.5) / len(self.chunk_values))
            cols[2].append((overlap_lookup[float(o)] + 0.5) / len(self.overlap_values))
            rgrid = self.rerank_values_for(k)
            rlookup = {int(v): i for i, v in enumerate(rgrid)}
            cols[3].append((rlookup[r] + 0.5) / len(rgrid))
            cols[4].append((context_lookup[float(b)] + 0.5) / len(self.context_values))
        return np.column_stack(cols)

    def key(self, row: np.ndarray | tuple) -> tuple[int, int, int, int, int]:
        a = np.asarray(row)
        return (
            int(a[0]),
            int(a[1]),
            int(round(float(a[2]) * 100)),
            int(a[3]),
            int(a[4]),
        )

    def keys(self, configurations: np.ndarray) -> list[tuple[int, int, int, int, int]]:
        return [self.key(row) for row in np.asarray(configurations)]

    def dataframe(self, configurations: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(np.asarray(configurations), columns=CONFIG_COLUMNS)
        return frame.astype(
            {
                "top_k": int,
                "chunk_tokens": int,
                "overlap": float,
                "rerank_depth": int,
                "context_tokens": int,
            }
        )

    def iter_all(self, batch_size: int = 4096):
        batch = []
        for k in self.top_k_values:
            for values in product(
                (k,),
                self.chunk_values,
                self.overlap_values,
                self.rerank_values_for(k),
                self.context_values,
            ):
                batch.append(values)
                if len(batch) >= batch_size:
                    yield np.asarray(batch, dtype=float)
                    batch = []
        if batch:
            yield np.asarray(batch, dtype=float)
