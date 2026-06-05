
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .knowledge_processor import EmbeddingManager

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "rag_config.json"

class RagPipeline:
    """RAG preparation pipeline: validate config/docs, chunk, and embed text."""

    def __init__(
        self,
        config_path: Path = _DEFAULT_CONFIG,
        embedding_manager: Optional[EmbeddingManager] = None,
    ) -> None:
        self.config_path = config_path
        self.embedding_manager = embedding_manager or EmbeddingManager()

    # -----------------------------------------------------------------------
    # Config & documents
    # -----------------------------------------------------------------------
    def load_config(self, path: Path) -> dict[str, Any]:
        """Load RAG JSON config from ``path`` into a dict.

        Raises ``RuntimeError`` if the file cannot be read or is not UTF-8.
        Raises ``ValueError`` if the content is not a JSON object.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"RAG config could not be read: {path}") from exc
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"RAG config must be UTF-8 text: {path}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"RAG config is not valid JSON: {path}") from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"RAG config root must be a JSON object, not {type(data).__name__}: {path}"
            )
        return data

    def load_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure every item is a dict; raise ``ValueError`` otherwise."""
        for i, item in enumerate(documents):
            if not isinstance(item, dict):
                raise ValueError(f"Document {i} must be a JSON object")
        return documents

    def document_text_and_metadata(
        self,
        doc: dict[str, Any],
        text_key: str,
        metadata_key: str,
    ) -> tuple[str, dict[str, Any]]:
        """Return body text for ``text_key`` and merged metadata.

        Merge order: if ``doc[metadata_key]`` is a dict, its key-value pairs are
        applied first. Then every other top-level key on ``doc`` (except
        ``text_key`` and ``metadata_key``) is copied into the result. If a key
        appears both in nested metadata and at the top level of ``doc``, the
        top-level value wins.
        """
        raw = doc.get(text_key)
        if raw is None:
            body = ""
        elif isinstance(raw, str):
            body = raw.strip()
        else:
            raise ValueError(
                f"{text_key!r} must be a string or null/omitted, got {type(raw).__name__}"
            )
        meta: dict[str, Any] = {}
        if isinstance(doc.get(metadata_key), dict):
            meta.update(doc[metadata_key])
        for k, v in doc.items():
            if k not in (text_key, metadata_key):
                meta[k] = v
        return body, meta

    # -----------------------------------------------------------------------
    # Chunking & embeddings
    # -----------------------------------------------------------------------
    def extract_ingest_config(self, cfg: dict[str, Any]) -> tuple[str, str, int, int]:
        """Validate ``cfg`` shape for chunking/ingestion and return core keys/sizes."""
        if not isinstance(cfg, dict):
            raise ValueError("RAG config root must be a JSON object")
        for section in ("ingestion", "chunking"):
            if section not in cfg:
                raise ValueError(f"RAG config missing required section {section!r}")
            if not isinstance(cfg[section], dict):
                raise ValueError(f"RAG config section {section!r} must be a JSON object")
        ing = cfg["ingestion"]
        ch = cfg["chunking"]
        for key in ("text_key", "metadata_key"):
            if key not in ing:
                raise ValueError(f"RAG config ingestion missing required key {key!r}")
        for key in ("chunk_size", "chunk_overlap"):
            if key not in ch:
                raise ValueError(f"RAG config chunking missing required key {key!r}")
        tk, mk = ing["text_key"], ing["metadata_key"]
        if not isinstance(tk, str) or not isinstance(mk, str):
            raise ValueError("RAG config ingestion.text_key and metadata_key must be strings")
        try:
            csz, ov = int(ch["chunk_size"]), int(ch["chunk_overlap"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "RAG config chunking.chunk_size and chunking.chunk_overlap must be integer-like numbers"
            ) from exc
        self.validate_chunking_params(csz, ov)
        return tk, mk, csz, ov

    @staticmethod
    def _word_chunk_length(text: str) -> int:
        """Word count for splitters (whitespace-separated tokens)."""
        return len(text.split())

    def validate_chunking_params(self, chunk_size: int, overlap: int) -> None:
        """Require sane word-window settings; raise ``ValueError`` otherwise."""
        if chunk_size < 1:
            raise ValueError(f"chunking.chunk_size must be >= 1, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"chunking.chunk_overlap must be >= 0, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                "chunking.chunk_overlap must be strictly less than chunking.chunk_size "
                f"(got overlap={overlap}, chunk_size={chunk_size})"
            )

    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split ``text`` into chunks based on word-count windows."""
        if text is None:
            raise ValueError("chunk_text: text must not be None")
        if not isinstance(text, str):
            raise ValueError(f"chunk_text: text must be str, got {type(text).__name__}")
        if text == "":
            return []
        try:
            csz = int(chunk_size)
            ov = int(overlap)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "chunk_text: chunk_size and chunk_overlap must be integer-like numbers"
            ) from exc
        self.validate_chunking_params(csz, ov)
        if not text.split():
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=csz,
            chunk_overlap=ov,
            length_function=self._word_chunk_length,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)

    def make_embed_fn(self) -> Callable[[str], np.ndarray]:
        """Build a function that embeds one string as a ``(1, dim)`` float32 row."""

        def embed_row(s: str) -> np.ndarray:
            """Embed ``s`` to a single row vector."""
            if not isinstance(s, str):
                raise TypeError(f"embed_row: expected str, got {type(s).__name__}")
            first = self.embedding_manager.generate_embedding(s)
            if first is None:
                raise RuntimeError(
                    "EmbeddingManager returned no embedding. "
                    "Check model availability/configuration."
                )
            v = np.asarray(first, dtype=np.float32)
            if v.size == 0:
                raise RuntimeError("Embedding model produced an empty vector")
            return v.reshape(1, -1)

        return embed_row

    def ingest_chunks(
        self,
        docs: list[dict[str, Any]],
        cfg: dict[str, Any],
        embed_fn: Callable[[str], np.ndarray],
    ) -> list[dict[str, Any]]:
        """Chunk ``docs`` per ``cfg``, attach embeddings and doc/chunk indices."""
        tk, mk, csz, ov = self.extract_ingest_config(cfg)
        rag_chunk: list[dict[str, Any]] = []
        for i, doc in enumerate(docs):
            if not isinstance(doc, dict):
                raise ValueError(f"Document {i} must be a JSON object")
            body, meta = self.document_text_and_metadata(doc, tk, mk)
            for chunk_i, chunk in enumerate(self.chunk_text(body, csz, ov)):
                embedding = embed_fn(chunk)
                rag_chunk.append(
                    {
                        "text": chunk,
                        "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                        "metadata": {**meta, "doc_index": i, "chunk_index": chunk_i},
                    }
                )
        return rag_chunk

    # -----------------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------------
    def run(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """End-to-end RAG prep: default config, validate docs, embed chunk records."""
        cfg = self.load_config(self.config_path)
        self.extract_ingest_config(cfg)
        docs = self.load_documents(docs)
        embed_fn = self.make_embed_fn()
        return self.ingest_chunks(docs, cfg, embed_fn)


def run_rag(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility wrapper around :class:`RagPipeline`."""
    return RagPipeline().run(docs)
