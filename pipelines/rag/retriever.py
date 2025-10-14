from typing import List, Tuple, Dict, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from vectorstore.faiss_store import FaissVectorStore
from pipelines.ingest.embeding import Embedder

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

def _simple_tokenize(s: str) -> List[str]:
    return [t for t in s.lower().split() if t.isalnum() or any(c.isalpha() for c in t)]

def mmr(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    candidates: List[int],
    k: int,
    lambda_mult: float = 0.5,
) -> List[int]:
    selected = []
    candidate_set = set(candidates)
    if len(candidates) <= k:
        return list(candidates)
    sim_to_query = np.dot(doc_embs[candidates], query_emb.T).flatten()
    selected.append(candidates[int(np.argmax(sim_to_query))])
    candidate_set.remove(selected[0])
    while len(selected) < k and candidate_set:
        mmr_scores = []
        for c in list(candidate_set):
            sim_q = np.dot(doc_embs[c], query_emb.T).item()
            sim_d = max(np.dot(doc_embs[c], doc_embs[selected].T).flatten().tolist()) if selected else 0.0
            score = lambda_mult * sim_q - (1 - lambda_mult) * sim_d
            mmr_scores.append((score, c))
        chosen = sorted(mmr_scores, key=lambda x: x[0], reverse=True)[0][1]
        selected.append(chosen)
        candidate_set.remove(chosen)
    return selected

class HybridRetriever:
    def __init__(self, store: FaissVectorStore, embedder: Optional[Embedder] = None):
        self.store = store
        self.embedder = embedder or store.embedder
        # Build BM25 corpus once
        self._bm25_docs = [ _simple_tokenize(t) for t in self.store.texts ]
        self._bm25 = BM25Okapi(self._bm25_docs)

    def search(
        self,
        query: str,
        k: int = 8,
        bm25_k: int = 50,
        vec_k: int = 50,
        alpha: float = 0.5,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
        rerank_model: Optional[str] = None,
        rerank_top_n: int = 20,
    ) -> List[Tuple[float, Dict]]:
        # Vector candidates
        vec_hits = self.store.search(query, k=vec_k)
        vec_scores = {h[1]["id"]: float(h[0]) for h in vec_hits}
        # BM25 candidates
        bm_scores_list = self._bm25.get_scores(_simple_tokenize(query))
        top_bm_idx = np.argsort(bm_scores_list)[::-1][:bm25_k]
        bm_scores = { self.store.ids[i]: float(bm_scores_list[i]) for i in top_bm_idx }

        # Normalize and fuse
        all_ids = list(set(list(vec_scores.keys()) + list(bm_scores.keys())))
        v = np.array([vec_scores.get(i, 0.0) for i in all_ids])
        b = np.array([bm_scores.get(i, 0.0) for i in all_ids])
        # scale to 0-1
        v = (v - v.min()) / (np.ptp(v) + 1e-9) if v.size else v
        b = (b - b.min()) / (np.ptp(b) + 1e-9) if b.size else b
        fused = alpha * v + (1 - alpha) * b
        ranked_ids = [x for _, x in sorted(zip(fused.tolist(), all_ids), reverse=True)][:max(k, rerank_top_n)]

        # Optional MMR
        if use_mmr:
            # Precompute embs for ranked docs for MMR
            doc_embs = self.embedder.encode([self.store.texts[self.store.ids.index(i)] for i in ranked_ids])
            query_emb = self.embedder.encode([query])[0]
            idx_map = {i: p for p, i in enumerate(ranked_ids)}
            selected_local = mmr(query_emb, doc_embs, list(range(len(ranked_ids))), k=max(k, rerank_top_n), lambda_mult=mmr_lambda)
            ranked_ids = [ranked_ids[j] for j in selected_local]

        # Optional cross-encoder reranker
        if rerank_model and CrossEncoder is not None:
            ce = CrossEncoder(rerank_model)
            pairs = [(query, self.store.texts[self.store.ids.index(i)]) for i in ranked_ids[:rerank_top_n]]
            rerank_scores = ce.predict(pairs).tolist()
            reranked = [x for _, x in sorted(zip(rerank_scores, ranked_ids[:rerank_top_n]), reverse=True)]
            ranked_ids = reranked + ranked_ids[rerank_top_n:]

        # Final top-k
        out = []
        for i in ranked_ids[:k]:
            pos = self.store.ids.index(i)
            # Combine fused score with vector score for stability
            score = 0.7 * vec_scores.get(i, 0.0) + 0.3 * bm_scores.get(i, 0.0)
            out.append((float(score), {"id": i, "text": self.store.texts[pos], "metadata": self.store.metas[pos]}))
        return out