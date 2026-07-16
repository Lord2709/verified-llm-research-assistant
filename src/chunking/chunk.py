import logging
from pathlib import Path

import nltk
from transformers import AutoTokenizer

from src.schemas import CleanedDocument, Chunk

logger = logging.getLogger(__name__)

TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3")
TARGET_CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50


def count_tokens(text: str) -> int:
    """Count tokens the way BGE-M3 actually would."""
    return len(TOKENIZER.tokenize(text))


def split_into_sentences(text: str) -> list[str]:
    """Split cleaned_text into sentences using NLTK."""
    return nltk.sent_tokenize(text)


def pack_sentences_into_chunks(sentences: list[str]) -> list[str]:
    """Greedily pack sentences into ~TARGET_CHUNK_TOKENS windows, backing off
    to whole sentences for OVERLAP_TOKENS between consecutive chunks."""
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_token_count = 0

    for sentence in sentences:
        sentence_token_count = count_tokens(sentence)

        if current_token_count + sentence_token_count > TARGET_CHUNK_TOKENS and current_chunk:
            chunks.append(" ".join(current_chunk))

            # walk backward through the just-finished chunk, accumulating
            # whole sentences until reaching ~OVERLAP_TOKENS (a token target,
            # not a sentence count)
            overlap_sentences: list[str] = []
            overlap_token_count = 0
            for prev_sentence in reversed(current_chunk):
                if overlap_token_count >= OVERLAP_TOKENS:
                    break
                overlap_sentences.insert(0, prev_sentence)
                overlap_token_count += count_tokens(prev_sentence)

            current_chunk = overlap_sentences
            current_token_count = overlap_token_count

        if sentence_token_count > TARGET_CHUNK_TOKENS and not current_chunk:
            logger.warning(
                f"Sentence exceeds target chunk size ({sentence_token_count} tokens); "
                "keeping it as its own oversized chunk rather than dropping it."
            )
            chunks.append(sentence)
            continue

        current_chunk.append(sentence)
        current_token_count += sentence_token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def build_chunk(doc: CleanedDocument, chunk_text: str, chunk_index: int) -> Chunk:
    """Assemble a Chunk from one packed piece of text plus its parent document."""
    token_count = count_tokens(chunk_text)
    if token_count > 2000:
        logger.warning(
            f"Unusually large chunk ({token_count} tokens) in document {doc.id}: {doc.title}"
        )
        
    return Chunk(
        document_id=doc.id,
        chunk_index=chunk_index,
        chunk_text=chunk_text,
        token_count=count_tokens(chunk_text),
        title=doc.title,
        source_url=doc.source_url,
        content_hash=doc.content_hash,
        source_metadata=doc.source_metadata,
    )


def chunk_document(doc: CleanedDocument) -> list[Chunk]:
    """Full per-document pipeline: sentence-split -> pack -> build Chunk objects."""
    sentences = split_into_sentences(doc.cleaned_text)
    chunk_texts = pack_sentences_into_chunks(sentences)
    return [build_chunk(doc, chunk_text, i) for i, chunk_text in enumerate(chunk_texts)]


def append_chunks(output_path: Path, chunks: list[Chunk]) -> None:
    """Cheap path: append a new document's chunks without touching the rest of the file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def write_all_chunks(output_path: Path, chunks: list[Chunk]) -> None:
    """Expensive path: rewrite the entire file, used when replacing a changed document's chunks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def chunk_all_documents(
    input_path: Path = Path("data/cleaned_documents.jsonl"),
    output_path: Path = Path("data/chunks.jsonl"),
) -> list[Chunk]:
    """Orchestrator: read cleaned_documents.jsonl, skip documents whose
    content_hash hasn't changed since last chunking, (re)chunk the rest,
    replacing all prior chunks for a changed document_id at once. Writes
    incrementally so a crash mid-run doesn't lose progress already made."""
    all_chunks: list[Chunk] = []

    existing_chunks_by_doc: dict[str, list[Chunk]] = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = Chunk.model_validate_json(line)
                existing_chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

    if not input_path.exists():
        logger.warning(f"Input file not found: {input_path}")
        return all_chunks

    with open(input_path, "r", encoding="utf-8") as f:
        doc_lines = [line for line in f if line.strip()]

    for line in doc_lines:
        try:
            doc = CleanedDocument.model_validate_json(line)
        except Exception as e:
            logger.error(f"Failed to parse a cleaned document line: {e}")
            continue

        existing_chunks_for_doc = existing_chunks_by_doc.get(doc.id, [])

        if existing_chunks_for_doc and all(
            c.content_hash == doc.content_hash for c in existing_chunks_for_doc
        ):
            logger.info(f"Skipping unchanged document {doc.id}: {doc.title}")
            all_chunks.extend(existing_chunks_for_doc)
            continue

        try:
            new_chunks_for_doc = chunk_document(doc)
        except Exception as e:
            logger.error(f"Failed to chunk document {doc.id}: {doc.title}. Error: {e}")
            continue

        all_chunks.extend(new_chunks_for_doc)
        existing_chunks_by_doc[doc.id] = new_chunks_for_doc

        if existing_chunks_for_doc:
            flattened = [c for chunks in existing_chunks_by_doc.values() for c in chunks]
            write_all_chunks(output_path, flattened)
        else:
            append_chunks(output_path, new_chunks_for_doc)

        logger.info(f"Chunked document {doc.id}: {doc.title} -> {len(new_chunks_for_doc)} chunks")

    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks = chunk_all_documents()
    print(f"Total chunks: {len(chunks)}")