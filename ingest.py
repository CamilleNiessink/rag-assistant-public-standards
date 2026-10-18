"""Load PDFs from data/, chunk them, embed them and build a FAISS index."""
import json

import faiss
from pypdf import PdfReader
from tqdm import tqdm

from common import (CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, INDEX_DIR, INDEX_FILE,
                    META_FILE, embed_texts)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows, breaking on whitespace where possible."""
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start + size // 2:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def load_documents() -> list[dict]:
    records = []
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found in data/. Add some documents first.")
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
                records.append({
                    "doc": pdf.name,
                    "page": page_no,
                    "chunk_id": f"{pdf.stem}-p{page_no}-c{i}",
                    "text": chunk,
                })
    return records


def main():
    INDEX_DIR.mkdir(exist_ok=True)
    records = load_documents()
    print(f"{len(records)} chunks from {len(set(r['doc'] for r in records))} documents")

    vectors = embed_texts([r["text"] for r in tqdm(records, desc="embedding")])
    index = faiss.IndexFlatIP(vectors.shape[1])   # inner product on unit vectors = cosine
    index.add(vectors)

    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    print(f"Index written to {INDEX_FILE}")


if __name__ == "__main__":
    main()
