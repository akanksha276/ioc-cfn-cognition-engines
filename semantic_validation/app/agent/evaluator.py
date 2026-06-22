# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .config import SAVConfig
from .models import (
    FailureMode,
    PMType,
    SAVInput,
    SAVOutput,
    Severity,
    SeverityType,
    SMType,
    description_for,
)
from .prompts import PM_EVALUATION_PROMPT, SM_EVALUATION_PROMPT

logger = logging.getLogger(__name__)

# ── Type maps ─────────────────────────────────────────────────────────────────

_SM_TYPE_MAP: Dict[str, SMType] = {sm.name.replace("_", "-"): sm for sm in SMType}
_SM_TYPE_MAP.update({sm.value: sm for sm in SMType})
_SM_TYPE_MAP.update({
    "SM-0": SMType.SM_0, "SM-1": SMType.SM_1, "SM-2": SMType.SM_2,
    "SM-3": SMType.SM_3, "SM-4": SMType.SM_4, "SM-5": SMType.SM_5,
})

_PM_TYPE_MAP: Dict[str, PMType] = {pm.name.replace("_", "-"): pm for pm in PMType}
_PM_TYPE_MAP.update({pm.value: pm for pm in PMType})
_PM_TYPE_MAP.update({
    "PM-0": PMType.PM_0, "PM-1": PMType.PM_1, "PM-2": PMType.PM_2,
    "PM-3": PMType.PM_3, "PM-4": PMType.PM_4, "PM-5": PMType.PM_5,
    "PM-6": PMType.PM_6, "PM-7": PMType.PM_7,
})

_SM_TYPE_TO_THRESHOLD = {
    SMType.SM_1: ("sm1_high", "sm1_medium"),
    SMType.SM_2: ("sm2_high", "sm2_medium"),
    SMType.SM_3: ("sm3_high", "sm3_medium"),
    SMType.SM_4: ("sm4_high", "sm4_medium"),
    SMType.SM_5: ("sm5_high", "sm5_medium"),
}

_PM_TYPE_TO_THRESHOLD = {
    PMType.PM_1: ("pm1_high", "pm1_medium"),
    PMType.PM_2: ("pm2_high", "pm2_medium"),
    PMType.PM_3: ("pm3_high", "pm3_medium"),
    PMType.PM_4: ("pm4_high", "pm4_medium"),
    PMType.PM_5: ("pm5_high", "pm5_medium"),
    PMType.PM_6: ("pm6_high", "pm6_medium"),
    PMType.PM_7: ("pm7_high", "pm7_medium"),
}


# ── Parsing ──────────────────────────────────────────────────────────────────

def _safe_json_parse(raw: str) -> Optional[Dict[str, Any]]:
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        pass
    return None


def _parse_sm_type(raw: str) -> SMType:
    return _SM_TYPE_MAP.get(str(raw).strip(), SMType.SM_0)


def _parse_pm_type(raw: str) -> PMType:
    return _PM_TYPE_MAP.get(str(raw).strip(), PMType.PM_0)


# ── Input validation ─────────────────────────────────────────────────────────

def _has_sm_input(sav_input: SAVInput) -> bool:
    """SM evaluation requires final_decision AND interaction_history.
    Without history, SM cannot tell whether the final decision is grounded
    in the agents' actual reasoning."""
    return bool(sav_input.final_decision) and bool(sav_input.interaction_history)


def _has_pm_input(sav_input: SAVInput) -> bool:
    """PM evaluation requires a latest_message to evaluate.
    interaction_history is optional context."""
    return bool(sav_input.latest_message)


# ── Evaluator ────────────────────────────────────────────────────────────────

class SAVEvaluator:
    def __init__(
        self,
        llm_provider: Callable[[str], str],
        config: Optional[SAVConfig] = None,
    ) -> None:
        self._llm = llm_provider
        self._cfg = config or SAVConfig()

    def evaluate(self, sav_input: SAVInput) -> SAVOutput:
        sm_enabled = _has_sm_input(sav_input)
        pm_enabled = _has_pm_input(sav_input)

        if not sm_enabled and not pm_enabled:
            logger.info(
                "SAVEvaluator: no final_decision / latest_message / "
                "interaction_history — skipping evaluation"
            )
            return SAVOutput()

        # Run SM and PM concurrently when both apply
        sm_modes: List[FailureMode] = []
        pm_modes: List[FailureMode] = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            sm_future = pool.submit(self._evaluate_sm, sav_input) if sm_enabled else None
            pm_future = pool.submit(self._evaluate_pm, sav_input) if pm_enabled else None

            if sm_future is not None:
                sm_modes = sm_future.result()
            if pm_future is not None:
                pm_modes = pm_future.result()

        return SAVOutput(failure_modes=sm_modes + pm_modes)

    # ── SM evaluation ────────────────────────────────────────────────────────

    def _evaluate_sm(self, sav_input: SAVInput) -> List[FailureMode]:
        try:
            raw = self._llm(self._build_sm_prompt(sav_input))
            parsed = _safe_json_parse(raw)
            if parsed is None:
                logger.warning("SAVEvaluator: failed to parse SM LLM response")
                return []
            return self._build_failure_modes(parsed, kind="sm")
        except Exception:
            logger.warning("SAVEvaluator: SM LLM call failed", exc_info=True)
            return []

    def _build_sm_prompt(self, sav_input: SAVInput) -> str:
        participants_str = json.dumps(
            [{"id": p.id, "name": p.name, "role": p.role} for p in sav_input.participants],
            indent=2,
        )
        history = list(sav_input.interaction_history or [])
        if sav_input.latest_message and (not history or history[-1] != sav_input.latest_message):
            history.append(sav_input.latest_message)
        history_str = json.dumps(history, indent=2) if history else "(not provided)"
        return (
            SM_EVALUATION_PROMPT
            .replace("{mission}", sav_input.mission)
            .replace("{context}", sav_input.context or "(not provided)")
            .replace("{participants}", participants_str)
            .replace("{final_decision}", sav_input.final_decision or "(not provided)")
            .replace("{interaction_history}", history_str)
        )

    # ── PM evaluation ────────────────────────────────────────────────────────

    def _evaluate_pm(self, sav_input: SAVInput) -> List[FailureMode]:
        try:
            raw = self._llm(self._build_pm_prompt(sav_input))
            parsed = _safe_json_parse(raw)
            if parsed is None:
                logger.warning("SAVEvaluator: failed to parse PM LLM response")
                return []
            return self._build_failure_modes(parsed, kind="pm")
        except Exception:
            logger.warning("SAVEvaluator: PM LLM call failed", exc_info=True)
            return []

    def _build_pm_prompt(self, sav_input: SAVInput) -> str:
        participants_str = json.dumps(
            [{"id": p.id, "name": p.name, "role": p.role} for p in sav_input.participants],
            indent=2,
        )
        history_str = (
            json.dumps(sav_input.interaction_history, indent=2)
            if sav_input.interaction_history
            else "(not provided)"
        )
        return (
            PM_EVALUATION_PROMPT
            .replace("{mission}", sav_input.mission)
            .replace("{context}", sav_input.context or "(not provided)")
            .replace("{participants}", participants_str)
            .replace("{latest_message}", sav_input.latest_message or "(not provided)")
            .replace("{interaction_history}", history_str)
        )

    # ── Severity / FailureMode construction ──────────────────────────────────

    def _severity_for(
        self,
        fm_type: Union[SMType, PMType],
    ) -> Tuple[float, float]:
        if isinstance(fm_type, SMType):
            threshold_keys = _SM_TYPE_TO_THRESHOLD.get(fm_type)
        else:
            threshold_keys = _PM_TYPE_TO_THRESHOLD.get(fm_type)

        if threshold_keys:
            high = getattr(self._cfg, threshold_keys[0])
            medium = getattr(self._cfg, threshold_keys[1])
        else:
            high = self._cfg.default_high
            medium = self._cfg.default_medium
        return high, medium

    def _build_failure_modes(
        self,
        parsed: Dict[str, Any],
        kind: str,
    ) -> List[FailureMode]:
        failure_modes: List[FailureMode] = []
        for fm in parsed.get("failure_modes") or []:
            raw_type = fm.get("type", "SM-0" if kind == "sm" else "PM-0")
            if kind == "sm":
                fm_type: Union[SMType, PMType] = _parse_sm_type(raw_type)
            else:
                fm_type = _parse_pm_type(raw_type)

            score = float(fm.get("score", 0.5))
            high, medium = self._severity_for(fm_type)
            severity = Severity(type=SeverityType.LOW, high=high, medium=medium)
            severity.type = severity.evaluate(score)

            failure_modes.append(FailureMode(
                type=fm_type,
                score=score,
                severity=severity,
                description=description_for(fm_type),
                reasoning=str(fm.get("reasoning", "")),
            ))
        return failure_modes
