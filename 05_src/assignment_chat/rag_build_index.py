# rag_build_index.py
"""
Build a persistent ChromaDB index over selected course notebooks.

- Reads Jupyter notebooks from ./01_materials/lab (First three lab files)
- Extracts markdown cells
- Concatenates consecutive markdown cells into ~800–1000 char chunks
- Embeds chunks with OpenAI embeddings
- Stores them in a persistent Chroma collection: "course_materials"
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from openai_client import get_client
from dotenv import load_dotenv
from pathlib import Path


# ---------- CONFIG ----------

# This file is in ./05_src/assignment_chat
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR.parent / ".secrets")

# Project root: one level up from 05_src
REPO_ROOT = BASE_DIR.parent.parent  # deploy-ai/

# Notebooks directory (as you specified)
LAB_DIR = REPO_ROOT / "01_materials" / "labs"

# Specific notebook filenames to index
NOTEBOOK_FILES = [
    "01_1_introduction.ipynb",
    "01_2_longer_context.ipynb",
    "01_3_local_model.ipynb",
]

# Chroma persistence directory (will be created if it doesn't exist)
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "course_materials"

# Embedding model name used in the course (adjust if needed)
EMBED_MODEL = "text-embedding-3-small"

# ---------- HELPERS ----------

def load_notebook_markdown_cells(nb_path: Path) -> List[str]:
    """
    Load a Jupyter notebook and return a list of markdown cell texts.
    """
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    markdown_texts: List[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            # 'source' is a list of lines; join them into a single string
            src = cell.get("source", [])
            text = "".join(src)
            # Strip whitespace-only cells
            if text.strip():
                markdown_texts.append(text)
    return markdown_texts

def chunk_markdown_cells(
    texts: List[str],
    target_chunk_size: int = 900,
    max_chunk_size: int = 1200,
) -> List[str]:
    """
    Concatenate consecutive markdown cells into chunks.

    - Start with an empty buffer.
    - Keep adding cells until ~target_chunk_size.
    - If adding a cell would exceed max_chunk_size, start a new chunk.
    """
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for cell_text in texts:
        cell_len = len(cell_text)

        # If empty buffer, always start with this cell
        if current_len == 0:
            current.append(cell_text)
            current_len = cell_len
            continue

        # If adding this cell stays within max_chunk_size, keep concatenating
        if current_len + cell_len <= max_chunk_size:
            current.append("\n\n" + cell_text)
            current_len += cell_len
        else:
            # Finish current chunk and start a new one
            chunks.append("".join(current))
            current = [cell_text]
            current_len = cell_len

    # Add any remaining text
    if current:
        chunks.append("".join(current))

    return chunks


def get_chroma_collection():
    """
    Create or load a persistent Chroma collection.
    """
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using OpenAI embeddings via the course API Gateway.
    """
    client = get_client()
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in resp.data]

# ---------- MAIN BUILD LOGIC ----------

def main():
    # Sanity check: list available notebooks
    notebook_paths: List[Path] = []
    for name in NOTEBOOK_FILES:
        path = LAB_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Notebook not found: {path}")
        notebook_paths.append(path)

    print("Notebooks to index:")
    for p in notebook_paths:
        print(" -", p)

    collection = get_chroma_collection()

    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []

    for nb_path in notebook_paths:
        nb_name = nb_path.name
        print(f"\nProcessing notebook: {nb_name}")

        markdown_cells = load_notebook_markdown_cells(nb_path)
        print(f"  Found {len(markdown_cells)} markdown cells")

        chunks = chunk_markdown_cells(markdown_cells)
        print(f"  Created {len(chunks)} chunks after concatenation")

        for idx, chunk in enumerate(chunks):
            doc_id = f"{nb_name}_chunk_{idx}"

            all_ids.append(doc_id)
            all_docs.append(chunk)
            all_metadatas.append(
                {
                    "notebook": nb_name,
                    "chunk_index": idx,
                    "source_path": str(nb_path),
                }
            )

    if not all_docs:
        print("No documents to index; exiting.")
        return

    print(f"\nEmbedding {len(all_docs)} chunks...")
    embeddings = embed_texts(all_docs)

    print("Upserting into Chroma collection...")
    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_docs,
        metadatas=all_metadatas,
    )

    print(
        f"Done. Indexed {len(all_docs)} chunks into collection "
        f"'{COLLECTION_NAME}' at '{CHROMA_DIR}'."
    )


if __name__ == "__main__":
    main()