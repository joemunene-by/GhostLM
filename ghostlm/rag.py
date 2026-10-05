"""Hybrid retriever over the RAG index built by scripts/build_rag_index.py.

Dense BGE search finds passages by meaning but is unreliable for exact
identifiers: "CVE-2021-44228" and "CVE-2021-45046" embed almost identically.
So identifiers in the query (CVE, CWE, CAPEC, ATT&CK technique IDs) are first
looked up exactly in an ID-to-chunk map, and dense results fill the rest.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

ID_PATTERN = re.compile(
    r"\b(CVE-\d{4}-\d{4,7}|CWE-\d{1,5}|CAPEC-\d{1,5}|T\d{4}(?:\.\d{3})?)\b",
    re.IGNORECASE,
)
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def extract_ids(text: str) -> List[str]:
    """Return the security identifiers in ``text``, uppercased, in order, deduplicated."""
    seen: Dict[str, None] = {}
    for m in ID_PATTERN.finditer(text):
        seen.setdefault(m.group(1).upper(), None)
    return list(seen)


def bge_embedder(name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu") -> Callable[[str], np.ndarray]:
    """Load a BGE bi-encoder and return a query -> L2-normalized vector function."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()

    def embed(query: str) -> np.ndarray:
        enc = tok(QUERY_INSTRUCTION + query, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model(**enc).last_hidden_state[:, 0]
        return F.normalize(emb, p=2, dim=-1).cpu().float().numpy().reshape(-1)

    return embed


class LazyChunks(Sequence):
    """Read-only view of chunks.jsonl that loads one record at a time by byte offset."""

    def __init__(self, path: Path, offsets: np.ndarray):
        self.path = path
        self.offsets = offsets
        self._fh = path.open("rb")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        self._fh.seek(int(self.offsets[i]))
        return json.loads(self._fh.readline())


def _scan_chunks(path: Path):
    """Byte offsets of every line plus the exact-ID index, in one streaming pass."""
    offsets, id_index, pos = [], {}, 0
    with path.open("rb") as f:
        for i, line in enumerate(f):
            offsets.append(pos)
            pos += len(line)
            chunk = json.loads(line)
            for ident in extract_ids(f"{chunk.get('ref', '')} {chunk.get('text', '')}"):
                id_index.setdefault(ident, []).append(i)
    return np.array(offsets, dtype=np.int64), id_index


class HybridRetriever:
    """Exact identifier lookup first, then dense cosine search."""

    def __init__(self, matrix: np.ndarray, chunks: Sequence,
                 embed: Optional[Callable[[str], np.ndarray]] = None,
                 id_index: Optional[Dict[str, List[int]]] = None):
        self.matrix = matrix
        self.chunks = chunks
        self._embed = embed
        if id_index is None:
            id_index = {}
            for i, chunk in enumerate(chunks):
                for ident in extract_ids(f"{chunk.get('ref', '')} {chunk.get('text', '')}"):
                    id_index.setdefault(ident, []).append(i)
        self.id_index = id_index

    @classmethod
    def load(cls, rag_dir: str | Path = "data/rag",
             embed: Optional[Callable[[str], np.ndarray]] = None) -> "HybridRetriever":
        """Open an index without reading it into RAM: the matrix is memory-mapped and
        passages are read on demand. Offsets and the ID index are cached beside it."""
        rag_dir = Path(rag_dir)
        chunks_path = rag_dir / "chunks.jsonl"
        matrix = np.load(rag_dir / "index.npy", mmap_mode="r")
        stamp = f"{chunks_path.stat().st_size}:{int(chunks_path.stat().st_mtime)}"
        cache, offsets_path = rag_dir / "id_index.json", rag_dir / "chunk_offsets.npy"
        cached = json.loads(cache.read_text()) if cache.exists() else {}
        if cached.get("stamp") == stamp and offsets_path.exists():
            offsets, id_index = np.load(offsets_path), cached["id_index"]
        else:
            offsets, id_index = _scan_chunks(chunks_path)
            np.save(offsets_path, offsets)
            cache.write_text(json.dumps({"stamp": stamp, "id_index": id_index}))
        return cls(matrix, LazyChunks(chunks_path, offsets), embed, id_index)

    @property
    def embed(self) -> Callable[[str], np.ndarray]:
        if self._embed is None:
            self._embed = bge_embedder()
        return self._embed

    def search(self, query: str, k: int = 4) -> List[dict]:
        """Return up to ``k`` passages as dicts with text, source, ref, score and match."""
        hits: List[dict] = []
        taken = set()

        for ident in extract_ids(query):
            # Chunks that name the ID in their ref are the record itself, not a mention.
            ranked = sorted(self.id_index.get(ident, []),
                            key=lambda i: ident not in str(self.chunks[i].get("ref", "")).upper())
            for i in ranked:
                if len(hits) >= k:
                    break
                if i not in taken:
                    taken.add(i)
                    hits.append(self._hit(i, 1.0, "id"))

        if len(hits) < k:
            scores = self.matrix @ self.embed(query)
            # Partial sort: only the top few of hundreds of thousands are needed.
            top = min(len(scores), k + len(taken))
            candidates = np.argpartition(-scores, top - 1)[:top]
            for i in candidates[np.argsort(-scores[candidates])]:
                if len(hits) >= k:
                    break
                i = int(i)
                if i not in taken:
                    taken.add(i)
                    hits.append(self._hit(i, float(scores[i]), "dense"))
        return hits

    def _hit(self, i: int, score: float, match: str) -> dict:
        c = self.chunks[i]
        return {"text": c.get("text", ""), "source": c.get("source", ""),
                "ref": c.get("ref", ""), "score": score, "match": match}
