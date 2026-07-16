import logging
from pathlib import Path

from src.embedding.embed import get_embedding_model, get_chroma_collection
from src.retrieval.bm25 import tokenize, load_bm25_index

logger = logging.getLogger(__name__)

RRF_K = 60
FETCH_K = 20  # how many results to pull from EACH method before fusing
TOP_K = 5     # how many fused results to return to the caller


def dense_search(query: str, model, collection, top_k: int = FETCH_K) -> list[str]:
    """Embed the query with BGE-M3 and query Chroma for the top_k nearest
    chunks by cosine similarity. Returns chunk_ids ranked best-first."""
    query_embedding = model.encode([query])
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["ids"][0]


def sparse_search(query: str, index, chunk_ids: list[str], top_k: int = FETCH_K) -> list[str]:
    """Tokenize the query and score it against the BM25 index. Returns
    chunk_ids ranked best-first (highest BM25 score first)."""
    tokenized_query = tokenize(query)
    scores = index.get_scores(tokenized_query)
    ranked = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk_id for chunk_id, _ in ranked[:top_k]]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Combine multiple ranked lists of chunk_ids into a single fused ranking.
    Each chunk earns 1/(k+rank) from every list it appears in (rank is
    1-indexed); a chunk that ranks well in both lists accumulates a higher
    combined score than one that only appears in one list."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def hybrid_search(query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """Orchestrator: run dense + sparse search, fuse via RRF, return the
    top_k fused (chunk_id, score) pairs."""
    model = get_embedding_model()
    collection = get_chroma_collection()
    bm25_index, bm25_chunk_ids = load_bm25_index()

    dense_ids = dense_search(query, model, collection)
    sparse_ids = sparse_search(query, bm25_index, bm25_chunk_ids)

    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])
    return fused[:top_k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_query = "GQA to MLA transition financial LLM"
    top_results = hybrid_search(test_query)
    for chunk_id, score in top_results:
        print(f"{chunk_id}: {score:.4f}")
