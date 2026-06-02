# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Base class for all Cognition Engines in the IoC CFN architecture.

A *Cognition Engine* is a self-contained reasoning unit that exposes a single
:meth:`CognitionEngine.run` entry-point.  Callers select an operation via the
engine's declared :class:`~enum.Enum` of actions and pass an arbitrary JSON-
compatible payload dict.

Model configuration
-------------------
Every engine requires a :class:`ModelConfig` at construction time.  The config
declares whether the engine uses a **remote LLM** (via litellm), a **local
model** (filesystem path), or both (e.g. remote LLM + local embedding model).

Remote LLM (required unless ``local_model_path`` covers all engine needs):
- ``llm_model``   — litellm model string, e.g. ``"openai/azure/gpt-4o"``.
- ``llm_api_key`` — API key for the provider (``None`` for key-less endpoints).
- ``llm_base_url`` — Base URL override (e.g. LiteLLM proxy); ``None`` = default.

Local model (optional):
- ``local_model_path`` — Absolute filesystem path to a local model directory
  (e.g. a HuggingFace model for embeddings).  When set, the path is validated
  at engine initialisation.

Concrete subclasses must:

1. Define an inner (or module-level) ``Action`` enum whose members name every
   operation the engine supports.
2. Implement :meth:`run` to dispatch on that enum.

Example skeleton::

    class MyEngine(CognitionEngine):
        class Action(str, Enum):
            DO_THING = "do_thing"

        @property
        def action_enum(self):
            return MyEngine.Action

        async def run(self, action, payload):
            if action == MyEngine.Action.DO_THING:
                ...
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


def handles(*actions: Any):
    """Decorator that registers a method as the handler for one or more actions.

    Usage::

        @handles(MyAction.DO_THING)
        async def _do_thing(self, payload): ...

        @handles(MyAction.INGEST, MyAction.EXTRACT_AND_INGEST)  # alias
        async def _ingest(self, payload): ...
    """

    def decorator(fn: Any) -> Any:
        fn._handles = actions
        return fn

    return decorator


@dataclass
class ModelConfig:
    """Model configuration for a :class:`CognitionEngine`.

    Attributes:
        llm_model: litellm model identifier
            (e.g. ``"openai/gpt-4o"``, ``"openai/azure/gpt-4o"``).
            Defaults to the ``LLM_MODEL`` environment variable or
            ``"openai/gpt-4o"`` when unset.
        llm_api_key: API key for the LLM provider.  Defaults to the
            ``LLM_API_KEY`` environment variable.  May be ``None`` for
            proxies / key-less endpoints.
        llm_base_url: Base URL override for the LLM provider (e.g. a
            LiteLLM proxy).  Defaults to the ``LLM_BASE_URL`` environment
            variable.  ``None`` means use the provider's default endpoint.
        local_model_path: Absolute path to a local model directory (e.g.
            ``granite-embedding-30m-english``).  When set, the path is
            checked to exist at engine initialisation.  ``None`` means no
            local model is used.
    """

    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "openai/gpt-4o")
    )
    llm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    llm_base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL")
    )
    local_model_path: Optional[str] = field(
        default_factory=lambda: os.getenv("LOCAL_MODEL_PATH")
    )

    # ── derived helpers ────────────────────────────────────────────────────

    @property
    def has_remote_llm(self) -> bool:
        """``True`` when a remote LLM model string is configured."""
        return bool(self.llm_model)

    @property
    def has_local_model(self) -> bool:
        """``True`` when a local model path is configured."""
        return bool(self.local_model_path)

    @property
    def litellm_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments to forward to litellm calls."""
        kwargs: Dict[str, Any] = {}
        if self.llm_api_key:
            kwargs["api_key"] = self.llm_api_key
        if self.llm_base_url:
            kwargs["base_url"] = self.llm_base_url
        return kwargs


class CognitionEngine(ABC):
    """Abstract base for all cognition engines.

    Subclasses declare their supported operations as an :class:`~enum.Enum`,
    tag each handler method with :func:`handles`, and the base class dispatches
    automatically — no ``run()`` override needed in subclasses.

    Args:
        model_config: Model configuration specifying which LLM or local
            model to use.  When ``None``, a default :class:`ModelConfig` is
            constructed from environment variables.

    Raises:
        ValueError: If ``model_config`` provides neither a remote LLM nor a
            local model path — i.e. the engine has no model to call.
        FileNotFoundError: If ``model_config.local_model_path`` is set but
            does not exist on the filesystem.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Collect all :func:`handles`-decorated methods into ``cls._handlers``."""
        super().__init_subclass__(**kwargs)
        # Start from any handlers already registered on a parent subclass.
        handlers: Dict[Any, str] = dict(getattr(cls, "_handlers", {}))
        for attr_name in vars(cls):
            method = vars(cls)[attr_name]
            if callable(method) and hasattr(method, "_handles"):
                for action in method._handles:
                    handlers[action] = attr_name
        cls._handlers: Dict[Any, str] = handlers

    def __init__(self, model_config: Optional[ModelConfig] = None) -> None:
        self.model_config: ModelConfig = model_config or ModelConfig()
        self._validate_model_config()
        logger.info(
            "%s initialised  llm_model=%r  local_model=%r",
            self.__class__.__name__,
            self.model_config.llm_model or "<none>",
            self.model_config.local_model_path or "<none>",
        )

    # ── abstract interface ─────────────────────────────────────────────────

    @property
    @abstractmethod
    def action_enum(self) -> Type[Enum]:
        """Return the :class:`~enum.Enum` class that enumerates this engine's actions."""

    async def run(self, action: Enum, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch *action* to the method registered with :func:`handles`.

        Args:
            action: A member of :attr:`action_enum`.
            payload: JSON-compatible dict containing all inputs required by the action.

        Returns:
            JSON-compatible dict with the operation result.

        Raises:
            ValueError: If *action* is not registered on this engine.
            NotImplementedError: If no handler is registered for *action*.
        """
        self._validate_action(action)
        handler_name = self._handlers.get(action)
        if handler_name is None:  # pragma: no cover
            raise NotImplementedError(f"No handler registered for {action!r}")
        return await getattr(self, handler_name)(payload)

    # ── discoverability ───────────────────────────────────────────────────

    def describe_actions(self) -> Dict[str, Any]:
        """Return a description of every action registered on this engine.

        Returns a dict keyed by action value (str) with the fields:

        - ``action`` — the enum member name (e.g. ``"EXTRACT"``)
        - ``value`` — the string value callers pass (e.g. ``"extract"``)
        - ``handler`` — name of the method that handles it
        - ``description`` — first line of the handler's docstring, or ``None``

        Example::

            engine.describe_actions()
            # {
            #   "extract": {"action": "EXTRACT", "value": "extract",
            #                "handler": "_extract", "description": "Extract concepts ..."},
            #   ...
            # }
        """
        result: Dict[str, Any] = {}
        for action, handler_name in self._handlers.items():
            method = getattr(self, handler_name, None)
            doc = getattr(method, "__doc__", None)
            first_line = doc.strip().splitlines()[0] if doc else None
            result[action.value] = {
                "action": action.name,
                "value": action.value,
                "handler": handler_name,
                "description": first_line,
            }
        return result

    # ── validation ─────────────────────────────────────────────────────────

    def _validate_model_config(self) -> None:
        """Validate that the engine has at least one usable model source.

        Raises:
            ValueError: Neither remote LLM nor local model path is configured.
            FileNotFoundError: ``local_model_path`` is set but does not exist.
        """
        cfg = self.model_config

        if not cfg.has_remote_llm and not cfg.has_local_model:
            raise ValueError(
                f"{self.__class__.__name__}: ModelConfig must specify at least one of "
                "'llm_model' (remote LLM) or 'local_model_path' (local model). "
                "Set LLM_MODEL or LOCAL_MODEL_PATH environment variables, or pass "
                "a ModelConfig explicitly."
            )

        if cfg.local_model_path:
            p = Path(cfg.local_model_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"{self.__class__.__name__}: local_model_path {str(p)!r} does not exist. "
                    "Ensure the model is downloaded or update LOCAL_MODEL_PATH."
                )

    def _validate_action(self, action: Enum) -> None:
        """Raise :class:`ValueError` if *action* is not registered on this engine."""
        if action not in self._handlers:
            raise ValueError(
                f"{action!r} is not a valid action for {self.__class__.__name__}. "
                f"Expected one of: {[a.value for a in self.action_enum]}"
            )
