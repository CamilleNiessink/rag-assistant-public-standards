"""Shared configuration and helpers."""
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
INDEX_FILE = INDEX_DIR / "chunks.faiss"
META_FILE = INDEX_DIR / "chunks.json"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 800        # characters
CHUNK_OVERLAP = 250     # characters
TOP_K = 6
MIN_SCORE = 0.33        # cosine similarity below this -> "not found"

client = OpenAI()


def embed_texts(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Embed a list of strings, returning a (n, d) float32 array, L2-normalised."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    arr = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def load_meta() -> list[dict]:
    with open(META_FILE, encoding="utf-8") as f:
        return json.load(f)
