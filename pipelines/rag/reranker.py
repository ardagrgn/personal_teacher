# a reranker function will add if it seems useful

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    from sentence_transformers import CrossEncoder
except Exception as e:
    CrossEncoder = None  # type: ignore


@dataclass
class ScoredPassage:
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float  # reranker score (higher is better)
    base_score: Optional[float] = None  # original retriever score


def _to_passages(
    candidates: Iterable[Union[Tuple[float, Dict[str, Any]], Dict[str, Any]]]
) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[Optional[float]]]:
    """
    Normalizes candidate inputs to parallel arrays.
    Supports:
      - List[Tuple[score, { 'id','text','metadata': {...} }]]
      - List[{ 'id','text','metadata': {...}, 'score': optional }]
    """
    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    base_scores: List[Optional[float]] = []

    for item in candidates:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
            s, d = item
            ids.append(str(d.get("id", "")))
            texts.append(str(d.get("text", "")))
            metas.append(dict(d.get("metadata", {})))
            base_scores.append(float(s))
        elif isinstance(item, dict):
            ids.append(str(item.get("id", "")))
            texts.append(str(item.get("text", "")))
            metas.append(dict(item.get("metadata", {})))
            base_scores.append(float(item.get("score"))) if "score" in item else base_scores.append(None)
        else:
            raise TypeError("Unsupported candidate format.")
    return ids, texts, metas, base_scores


class BaseReranker:
    def rerank(
        self,
        query: str,
        candidates: Iterable[Union[Tuple[float, Dict[str, Any]], Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[ScoredPassage]:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker using sentence-transformers.
    Default model: 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,  # e.g., 'cuda', 'cuda:0', 'cpu'
        batch_size: int = 32,
        fuse_with_base: bool = True,
        alpha: float = 0.8,  # weight for CE score in fusion: final = alpha*ce + (1-alpha)*base
    ):
        if CrossEncoder is None:
            raise ImportError("sentence-transformers not installed. pip install sentence-transformers")
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size
        self.fuse_with_base = fuse_with_base
        self.alpha = float(alpha)

    def rerank(
        self,
        query: str,
        candidates: Iterable[Union[Tuple[float, Dict[str, Any]], Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[ScoredPassage]:
        ids, texts, metas, base_scores = _to_passages(candidates)
        if not texts:
            return []

        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size).tolist()

        # Optional fusion with base retriever scores
        fused: List[float] = []
        if self.fuse_with_base and any(s is not None for s in base_scores):
            # min-max normalize both streams for stability
            import math

            def _norm(arr: List[float]) -> List[float]:
                lo = min(arr)
                hi = max(arr)
                rng = hi - lo
                if math.isclose(rng, 0.0):
                    return [0.5 for _ in arr]
                return [(x - lo) / rng for x in arr]

            ce_n = _norm(scores)
            base_safe = [float(s if s is not None else 0.0) for s in base_scores]
            base_n = _norm(base_safe)
            fused = [self.alpha * c + (1.0 - self.alpha) * b for c, b in zip(ce_n, base_n)]
        else:
            fused = scores

        order = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)
        if top_k is not None:
            order = order[:top_k]

        out: List[ScoredPassage] = []
        for i in order:
            out.append(
                ScoredPassage(
                    id=ids[i],
                    text=texts[i],
                    metadata=metas[i],
                    score=float(fused[i]),
                    base_score=float(base_scores[i]) if base_scores[i] is not None else None,
                )
            )
        return out


class NoopReranker(BaseReranker):
    """
    Pass-through reranker that just sorts by base score (if provided).
    """

    def rerank(
        self,
        query: str,
        candidates: Iterable[Union[Tuple[float, Dict[str, Any]], Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[ScoredPassage]:
        ids, texts, metas, base_scores = _to_passages(candidates)
        idxs = list(range(len(ids)))
        if any(s is not None for s in base_scores):
            idxs.sort(key=lambda i: float(base_scores[i] or 0.0), reverse=True)
        if top_k is not None:
            idxs = idxs[:top_k]
        return [
            ScoredPassage(
                id=ids[i],
                text=texts[i],
                metadata=metas[i],
                score=float(base_scores[i] or 0.0),
                base_score=float(base_scores[i] or 0.0),
            )
            for i in idxs
        ]


# Convenience factory
def create_reranker(
    model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: Optional[str] = None,
    batch_size: int = 32,
    fuse_with_base: bool = True,
    alpha: float = 0.8,
) -> BaseReranker:
    if model is None or model.lower() in {"none", "noop"}:
        return NoopReranker()
    return CrossEncoderReranker(
        model_name=model,
        device=device,
        batch_size=batch_size,
        fuse_with_base=fuse_with_base,
        alpha=alpha,
    )