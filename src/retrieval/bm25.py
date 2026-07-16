import logging
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.schemas import Chunk

logger = logging.getLogger(__name__)

BM25_INDEX_PATH = Path("data/bm25_index.pkl")


def tokenize(text: str) -> list[str]:
    """Simple word-level tokenization for BM25 (lowercase, split on whitespace)."""
    return text.lower().split()


def load_chunks(input_path: Path = Path("data/chunks.jsonl")) -> list[Chunk]:
    """Read chunks.jsonl into a list of Chunk objects."""
    chunks_list: list[Chunk] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = Chunk.model_validate_json(line)
            chunks_list.append(chunk)
    return chunks_list


def build_bm25_index(chunks: list[Chunk]) -> tuple[BM25Okapi, list[str]]:
    """Tokenize all chunk texts and build a BM25Okapi index. Returns the index
    plus the ordered list of chunk_ids (BM25Okapi returns scores by position,
    so we need this to map scores back to actual chunks)."""
    tokenized_texts = [tokenize(chunk.chunk_text) for chunk in chunks]
    index = BM25Okapi(tokenized_texts)
    chunk_ids = [chunk.id for chunk in chunks]
    return index, chunk_ids


def save_bm25_index(index: BM25Okapi, chunk_ids: list[str], output_path: Path = BM25_INDEX_PATH) -> None:
    """Pickle the index + chunk_ids together to disk."""
    with open(output_path, "wb") as f:
        pickle.dump((index, chunk_ids), f)


def load_bm25_index(input_path: Path = BM25_INDEX_PATH) -> tuple[BM25Okapi, list[str]]:
    """Load a previously pickled index + chunk_ids from disk."""
    with open(input_path, "rb") as f:
        return pickle.load(f)


def build_and_save_bm25_index(
    chunks_path: Path = Path("data/chunks.jsonl"),
    index_path: Path = BM25_INDEX_PATH,
) -> None:
    """Orchestrator: load chunks, build the BM25 index, persist it."""
    chunks = load_chunks(chunks_path)
    index, chunk_ids = build_bm25_index(chunks)
    save_bm25_index(index, chunk_ids, index_path)
    logger.info(f"BM25 index built and saved to {index_path} with {len(chunks)} chunks.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_and_save_bm25_index()