import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.schemas import Chunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
CHROMA_PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "arxiv_chunks"
BATCH_SIZE = 16


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


def get_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformers wrapper around BGE-M3."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_chroma_collection() -> chromadb.Collection:
    """Get (or create) a persistent Chroma collection on disk, using cosine
    similarity since that's what BGE-M3 is designed/benchmarked around."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_existing_ids_and_hashes(collection: chromadb.Collection) -> dict[str, str]:
    """Return {chunk_id: content_hash} for everything already stored, so we
    can skip re-embedding chunks whose parent document hasn't changed."""
    result = collection.get(include=["metadatas"])
    return {
        chunk_id: metadata.get("content_hash", "")
        for chunk_id, metadata in zip(result["ids"], result["metadatas"])
    }


def embed_and_store_chunks(
    chunks: list[Chunk], model: SentenceTransformer, collection: chromadb.Collection
) -> None:
    """Embed the given chunks in batches and upsert them into the collection,
    storing chunk_text, embedding, and citation metadata (document_id,
    chunk_index, title, source_url, content_hash) together."""
    all_chunk_ids = [chunk.id for chunk in chunks]
    all_chunk_texts = [chunk.chunk_text for chunk in chunks]
    all_metadatas = [
        {
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "title": chunk.title,
            "source_url": chunk.source_url,
            "content_hash": chunk.content_hash,
        }
        for chunk in chunks
    ]
    embeddings = model.encode(all_chunk_texts)
    collection.upsert(
        ids=all_chunk_ids,
        documents=all_chunk_texts,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )


def embed_all_chunks(input_path: Path = Path("data/chunks.jsonl")) -> None:
    """Orchestrator: load chunks, skip ones already embedded with an
    unchanged content_hash, embed + store the rest in batches, log-and-continue
    on per-batch failure."""
    chunks = load_chunks(input_path)
    model = get_embedding_model()
    collection = get_chroma_collection()
    existing_ids_and_hashes = get_existing_ids_and_hashes(collection)

    filtered_chunks = [
        chunk for chunk in chunks
        if existing_ids_and_hashes.get(chunk.id) != chunk.content_hash
    ]
    logger.info(f"{len(filtered_chunks)} of {len(chunks)} chunks need embedding")

    for i in range(0, len(filtered_chunks), BATCH_SIZE):
        batch = filtered_chunks[i : i + BATCH_SIZE]
        try:
            embed_and_store_chunks(batch, model, collection)
            logger.info(f"Embedded batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")
        except Exception as e:
            logger.error(
                f"Failed to embed batch {i // BATCH_SIZE + 1} "
                f"(chunk ids {[c.id for c in batch]}): {e}"
            )
            continue


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embed_all_chunks()