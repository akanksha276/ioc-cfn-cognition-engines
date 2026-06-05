# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import numpy as np
import yaml
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


def _register_granite_in_fastembed(model_name: str, local_model_path: str) -> None:
    """Inject a custom model entry into fastembed's runtime ONNX registry.

    fastembed validates model_name against its internal list even when
    specific_model_path is provided.  This inserts a minimal DenseModelDescription
    so the real IBM Granite model name is accepted without using a proxy entry.
    The list mutation is process-scoped and does not touch any installed files.
    """
    from fastembed.common.model_description import DenseModelDescription, ModelSource
    from fastembed.text.onnx_embedding import supported_onnx_models

    if any(m.model == model_name for m in supported_onnx_models):
        return  # already registered

    entry = DenseModelDescription(
        model=model_name,
        dim=384,
        description="IBM Granite 30M English embedding model (local ONNX int8)",
        license="apache-2.0",
        size_in_GB=0.03,
        sources=ModelSource(hf=model_name),
        model_file="model_optimized.onnx",
    )
    supported_onnx_models.append(entry)
    logger.debug("Registered %s in fastembed ONNX registry", model_name)


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


class EmbeddingManager:
    def __init__(self, config_path=None):
        if config_path is None:
            # Use path relative to this file's location
            config_path = os.path.join(
                os.path.dirname(__file__), "embeddings_config.yml"
            )
        self.config = load_config(config_path)
        self.model_type = self.config.get("embedding_model_type") or "huggingface"
        self.model_name = (
            self.config.get("embedding_model_name")
            or "ibm-granite/granite-embedding-30m-english"
        )

        # Prefer local model path (e.g. from repo / Docker) over Hugging Face
        local_model_path = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
        if not local_model_path:
            # Fallback: repo root granite-embedding-30m-english (evidence/app/agent -> repo root)
            try:
                from pathlib import Path

                _repo_root = Path(__file__).resolve().parent.parent.parent.parent
                _default = _repo_root / "granite-embedding-30m-english"
                if _default.is_dir():
                    local_model_path = str(_default)
            except Exception:
                pass
        if local_model_path and os.path.isdir(local_model_path):
            logger.info("Loading embedding model from local path: %s", local_model_path)
            self.model_type = "huggingface"
            # Register the granite model in fastembed's runtime registry so we can
            # pass the real model name instead of a proxy entry.
            _register_granite_in_fastembed(self.model_name, local_model_path)
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.model = TextEmbedding(
                    model_name=self.model_name,
                    specific_model_path=local_model_path,
                )
        elif self.model_type == "huggingface":
            # fastembed is ONNX-based; no PyTorch/CUDA required.
            cache_dir = os.getenv("FASTEMBED_CACHE_PATH", "/tmp/fastembed_cache")
            self.model = TextEmbedding(model_name=self.model_name, cache_dir=cache_dir)

        # fallback to fastembed if no model type is configured or unknown type
        elif not self.model_type or self.model_type not in ("huggingface",):
            self.model_type = "huggingface"
            cache_dir = os.getenv("FASTEMBED_CACHE_PATH", "/tmp/fastembed_cache")
            self.model = TextEmbedding(model_name=self.model_name, cache_dir=cache_dir)

    def preprocess_text(self, text, chunk_size=512, overlap=50):
        # Character length chunking
        # TODO : change method based on text type
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])

            start += chunk_size - overlap

        return chunks

    def generate_embeddings(self, text_chunks):
        # fastembed.embed() returns a generator of numpy arrays
        return np.array(list(self.model.embed(text_chunks)))


if __name__ == "__main__":
    # Example test text
    test_text = "Collective intelligence: A trusted knowledge fabric for multi agent human societies"

    # Initialize EmbeddingManager with default config or specify a config path
    embedding_manager = EmbeddingManager()

    # Preprocess the text into chunks (optional, here just one chunk)
    text_chunks = embedding_manager.preprocess_text(test_text)

    # Generate embeddings for the chunks
    embeddings = embedding_manager.generate_embeddings(text_chunks)

    logger.info("Generated %d embeddings.", len(embeddings))
    logger.info("First embedding vector (truncated): %s", embeddings[0][:5])
