"""Bake model weights into the Docker image (NFR-6).

Run at image build time: after this, container start performs no HuggingFace
network access.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from mc.config import DENSE_MODEL_ID, SPARSE_LANGUAGE, SPARSE_MODEL_ID


def main() -> None:
    print(f"[bake] dense model: {DENSE_MODEL_ID}")
    SentenceTransformer(DENSE_MODEL_ID)

    print(f"[bake] sparse model: {SPARSE_MODEL_ID} ({SPARSE_LANGUAGE})")
    from fastembed import SparseTextEmbedding

    SparseTextEmbedding(model_name=SPARSE_MODEL_ID)

    print("[bake] done — weights cached")


if __name__ == "__main__":
    main()
