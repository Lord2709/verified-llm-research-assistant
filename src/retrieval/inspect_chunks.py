from pathlib import Path

from src.embedding.embed import load_chunks

CHUNK_IDS_TO_INSPECT = [
    "e0144f00-212a-4566-9e31-9684d4d56502",
    "84ae92bb-48b9-4a35-a7d6-eacf3a81222a",
    "e2a1dbee-076c-459c-a37d-3f9bf4628ca3",
    "da27cc1b-c415-4da8-9fe9-496d66084664",
]


def inspect_chunks(chunk_ids: list[str], chunks_path: Path = Path("data/chunks.jsonl")) -> None:
    chunks = load_chunks(chunks_path)
    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    for chunk_id in chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            print(f"{chunk_id}: NOT FOUND")
            continue
        preview = chunk.chunk_text[:200].replace("\n", " ")
        print(f"\n{chunk_id}")
        print(f"  title: {chunk.title}")
        print(f"  preview: {preview}...")


if __name__ == "__main__":
    inspect_chunks(CHUNK_IDS_TO_INSPECT)
