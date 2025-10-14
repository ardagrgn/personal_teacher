from typing import List, Dict
from vectorstore.faiss_store import FaissVectorStore
from pipelines.rag import HybridRetriever


SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer. "
    "If the answer is not in the context, say you don't know."
)

def format_context(hits) -> str:
    blocks = []
    for i, (_, h) in enumerate(hits, 1):
        title = h["metadata"].get("title", h["metadata"].get("path", h["metadata"].get("url", "")))
        blocks.append(f"[{i}] {title}\n{h['text']}")
    return "\n\n".join(blocks)

def retrieve(index_dir: str, query: str, k: int = 6):
    store = FaissVectorStore.load(index_dir)
    retriever = HybridRetriever(store)
    hits = retriever.search(query, k=k, bm25_k=50, vec_k=50, alpha=0.55, use_mmr=True, mmr_lambda=0.5)
    return hits

def build_prompt(query: str, hits) -> str:
    context = format_context(hits)
    prompt = f"{SYSTEM_PROMPT}\n\nQuery:\n{query}\n\nContext:\n{context}\n\nAnswer:"
    return prompt