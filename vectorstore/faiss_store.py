

import os, json
from typing import List, Dict, Sequence, Tuple, Optional
from pathlib import Path
from typing import Iterable, Union, List
import glob
from pathlib import Path
import numpy as np
import faiss
from pipelines.ingest.embeding import Embedder
from pipelines.ingest.load_funcs import load_many
from pipelines.ingest.chunking import chunk_docs


class FaissVectorStore:
    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder()
        self.index: Optional[faiss.Index] = None
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.metas: List[Dict] = []

    @property
    def dim(self) -> int:
        return self.embedder.dim

    def _ensure_index(self):
        if self.index is None:
            # Cosine via normalized embeddings => inner product
            self.index = faiss.IndexFlatIP(self.dim)

    def add(self, ids: Sequence[str], texts: Sequence[str], metas: Sequence[Dict]):
        assert len(ids) == len(texts) == len(metas)
        self._ensure_index()
        embs = self.embedder.encode(list(texts))
        self.index.add(embs)
        self.ids.extend(ids)
        self.texts.extend(texts)
        self.metas.extend(metas)

    def upsert(self, ids: Sequence[str], texts: Sequence[str], metas: Sequence[Dict]):
        # naive upsert: remove existing then add
        id_to_pos = {i: p for p, i in enumerate(self.ids)}
        keep_mask = np.ones(len(self.ids), dtype=bool)
        replace_positions = []
        for i in ids:
            if i in id_to_pos:
                keep_mask[id_to_pos[i]] = False
                replace_positions.append(id_to_pos[i])
        if not keep_mask.all():
            # rebuild with kept vectors
            kept_texts = [t for t, m in zip(self.texts, keep_mask) if m]
            kept_metas = [t for t, m in zip(self.metas, keep_mask) if m]
            kept_ids = [t for t, m in zip(self.ids, keep_mask) if m]
            embs = self.embedder.encode(kept_texts)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embs)
            self.ids, self.texts, self.metas = kept_ids, kept_texts, kept_metas
        self.add(ids, texts, metas)

    def save(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "index.faiss"))
        with open(out / "corpus.jsonl", "w", encoding="utf-8") as f:
            for i, t, m in zip(self.ids, self.texts, self.metas):
                f.write(json.dumps({"id": i, "text": t, "metadata": m}, ensure_ascii=False) + "\n")
        with open(out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": type(self.embedder.model).__name__,
                    "dim": self.dim,
                    "normalize": self.embedder.normalize,
                    "count": len(self.ids),
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, dir_path: str) -> "FaissVectorStore":
        out = Path(dir_path)
        with open(out / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        embedder = Embedder(model_name=manifest["model_name"], normalize=manifest.get("normalize", True))
        obj = cls(embedder=embedder)
        obj.index = faiss.read_index(str(out / "index.faiss"))
        with open(out / "corpus.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                obj.ids.append(rec["id"])
                obj.texts.append(rec["text"])
                obj.metas.append(rec["metadata"])
        return obj

    def search(self, query: str, k: int = 8) -> List[Tuple[float, Dict]]:
        q = self.embedder.encode([query])
        sims, idxs = self.index.search(q, k)
        out = []
        for score, i in zip(sims[0].tolist(), idxs[0].tolist()):
            if i == -1:
                continue
            out.append(
                (
                    float(score),
                    {"id": self.ids[i], "text": self.texts[i], "metadata": self.metas[i]},
                )
            )
        return out
    



def build_vectorstore(
    urls: Iterable[str] = (),
    files: Iterable[Union[str, Path]] = (),
    out_dir: Union[str, Path] = "vectorstore/faiss_index",
    max_tokens: int = 400,
    overlap_tokens: int = 40,
) -> str:
    # Expand globs for files
    expanded: List[str] = []
    for f in files or []:
        expanded.extend(glob.glob(str(f), recursive=True))
    docs = load_many(urls=urls or [], files=expanded or [])
    chunks = chunk_docs(docs, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    store = FaissVectorStore()
    store.add([c.id for c in chunks], [c.text for c in chunks], [c.metadata for c in chunks])
    out = Path(out_dir)
    store.save(out.as_posix())
    return out.as_posix()