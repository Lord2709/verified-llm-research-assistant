import logging
from pathlib import Path

from sentence_transformers import CrossEncoder

from src.embedding.embed import load_chunks
from src.retrieval.hybrid import hybrid_search

logger = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
FETCH_K = 20  # candidates pulled from hybrid_search before reranking narrows them down
TOP_K = 5     # final results returned after reranking


def get_reranker_model() -> CrossEncoder:
    """Load the BGE cross-encoder reranker."""
    return CrossEncoder(RERANKER_MODEL_NAME)


def get_chunk_texts(chunk_ids: list[str], chunks_path: Path = Path("data/chunks.jsonl")) -> dict[str, str]:
    """Load chunks.jsonl and return {chunk_id: chunk_text} for the given ids."""
    all_chunks = load_chunks(chunks_path)
    chunks_by_id = {chunk.id: chunk for chunk in all_chunks}
    return {
        chunk_id: chunks_by_id[chunk_id].chunk_text
        for chunk_id in chunk_ids
        if chunk_id in chunks_by_id
    }


def rerank(
    query: str,
    candidate_ids: list[str],
    chunk_texts_by_id: dict[str, str],
    model: CrossEncoder,
    top_k: int = TOP_K,
) -> list[tuple[str, float]]:
    """Score each (query, chunk_text) pair with the cross-encoder and return
    the top_k chunk_ids ranked by that relevance score, best first."""
    pairs = [(query, chunk_texts_by_id[chunk_id]) for chunk_id in candidate_ids if chunk_id in chunk_texts_by_id]
    scored_ids = [chunk_id for chunk_id in candidate_ids if chunk_id in chunk_texts_by_id]

    scores = model.predict(pairs)

    ranked = sorted(zip(scored_ids, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


def rerank_search(query: str, fetch_k: int = FETCH_K, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """Orchestrator: get a wide candidate set from hybrid_search, pull their
    chunk_text, rerank with the cross-encoder, return the final top_k."""
    fused_candidates = hybrid_search(query, top_k=fetch_k)
    candidate_ids = [chunk_id for chunk_id, _ in fused_candidates]

    chunk_texts_by_id = get_chunk_texts(candidate_ids)
    model = get_reranker_model()

    return rerank(query, candidate_ids, chunk_texts_by_id, model, top_k=top_k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_query = "GQA to MLA transition financial LLM"
    top_results = rerank_search(test_query)
    for chunk_id, score in top_results:
        print(f"{chunk_id}: {score:.4f}")
