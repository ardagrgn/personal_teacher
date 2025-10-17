from __future__ import annotations

import re
from typing import List, Dict, Iterable
from dataclasses import dataclass
from tqdm import tqdm




try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def _num_tokens(txt: str) -> int:
        return len(_ENC.encode(txt))
except Exception:
    def _num_tokens(txt: str) -> int:
        # Rough estimate if tiktoken not installed
        return max(1, int(len(txt) / 4))

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, str]

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

def _split_recursive(text: str, max_tokens: int, separators=_SEPARATORS) -> List[str]:
    if _num_tokens(text) <= max_tokens:
        return [text.strip()]
    for sep in separators:
        if sep and sep in text:
            parts = text.split(sep)
        else:
            parts = list(text)
        acc, cur = [], ""
        for p in parts:
            piece = (p if sep == "" else (p + sep))
            if _num_tokens(cur + piece) <= max_tokens:
                cur += piece
            else:
                if cur.strip():
                    acc.append(cur.strip())
                cur = piece
        if cur.strip():
            acc.append(cur.strip())
        if len(acc) > 1 or sep == "":
            out = []
            for a in acc:
                out.extend(_split_recursive(a, max_tokens, separators))
            return out
    return [text[:max_tokens]]  # fallback

def _apply_overlap(chunks: List[str], overlap_tokens: int, max_tokens: int) -> List[str]:
    if overlap_tokens <= 0 or len(chunks) <= 1:
        return [c.strip() for c in chunks if c.strip()]
    out, prev_tail = [], ""
    for c in chunks:
        merged = (prev_tail + ("\n\n" if prev_tail else "") + c).strip()
        # Trim to max_tokens
        if _num_tokens(merged) > max_tokens:
            # naive trim by characters until approx under limit
            while _num_tokens(merged) > max_tokens and len(merged) > 0:
                merged = merged[:-int(max(1, len(merged) * 0.05))]
        out.append(merged)
        # compute tail by tokens
        tokens = merged.split()
        prev_tail = " ".join(tokens[-overlap_tokens:]) if tokens else ""
    return out

def chunk_text(text: str, max_tokens: int = 400, overlap_tokens: int = 40) -> List[str]:
    base = _split_recursive(text, max_tokens=max_tokens)
    return _apply_overlap(base, overlap_tokens=overlap_tokens, max_tokens=max_tokens)

def chunk_docs(
    docs: Iterable,
    max_tokens: int = 400,
    overlap_tokens: int = 40,
    min_tokens: int = 20,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for d in tqdm(list(docs), desc="Chunking"):
        doc_hash = d.metadata.get("hash", "")
        parts = chunk_text(d.text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        for i, part in enumerate(parts):
            if _num_tokens(part) < min_tokens:
                continue
            md = dict(d.metadata)
            md.update({"chunk_index": str(i), "chunk_tokens": str(_num_tokens(part))})
            chunk_id = f"{doc_hash}:{i}" if doc_hash else f"anon:{id(part)}"
            chunks.append(Chunk(id=chunk_id, text=part, metadata=md))
    return chunks


