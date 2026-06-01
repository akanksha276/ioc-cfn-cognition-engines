# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""LLM-driven negotiation agent — persona and prompt configurable from YAML.

The agent receives the **complete** inbound ``SSTPNegotiateMessage`` and asks
an LLM to fill ``sao_response`` based on its private persona.  The LLM never
sees the agent's preferences directly — instead it receives a utility weight
table derived from ``prefer_low`` so it knows which options are more valuable.

Two prompt rendering modes are supported (``prompt_mode``):

* ``"sstp"`` *(default)* — the raw ``SSTPNegotiateMessage`` is embedded as a
  ``json`` fenced block.  Full protocol fidelity; best for models that handle
  structured JSON well.
* ``"english"`` — the same information is rendered as a plain-English narrative.
  Useful for models that handle prose better than raw JSON.

External config (YAML)
----------------------
::

    agents:
      - id: buyer
        type: llm
        prefer_low: true
        persona: >
          You are a cost-conscious procurement manager.
          Always push for the cheapest options early but accept
          mid-range options after round 10.
        prompt_mode: english     # optional, default "sstp"

Fallback behaviour
------------------
LLM JSON replies must include a ``reason`` string (1–3 sentences) alongside
``response`` and ``outcome``.  The server stores ``reason`` for all actions;
it does not change SAO outcomes.

When the LLM call fails or returns unparseable JSON the agent falls back to
a rule-based decision:

* **propose** → return the ideal offer (best outcome for ``prefer_low``).
* **respond** → reject (safe default that keeps the negotiation alive).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from protocol.sstp.negmas_sao import ResponseType, SAOResponse  # noqa

from .base_agent import BaseAgent, build_sstp_reply, _slug

# ── sys.path: ensure semantic_negotiation package is importable ─────────────
_sn_root = str(Path(__file__).resolve().parents[3])
if _sn_root not in sys.path:
    sys.path.insert(0, _sn_root)
from app.agent.reply_payload_utils import attach_reason, sanitize_reason  # noqa: E402


def _get_llm() -> Callable[[str], str]:
    """Lazy-load ``get_llm_provider`` so the agent can be instantiated without
    a configured LLM (tests, direct-env without LLM, etc.)."""
    try:
        from config.utils import get_llm_provider  # type: ignore

        return get_llm_provider()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "LLM provider not available. Ensure OPENAI_API_KEY (or equivalent) "
            "is set and the semantic_negotiation/app/config/utils.py is importable."
        ) from exc


class LLMAgent(BaseAgent):
    """Negotiation agent driven by an LLM with a configurable persona.

    The persona is a free-text description of the agent's negotiating style
    and objectives.  It is injected into every LLM prompt unchanged.

    Args:
        agent_id: Unique participant identifier.
        prefer_low: Determines which end of options lists is preferred
            (used to build the utility weight table shown to the LLM).
        persona: Free-text description of the agent's negotiating role and
            goals.  Injected verbatim into every prompt.
        prompt_mode: ``"sstp"`` (full JSON envelope) or ``"english"``
            (plain-language narrative).  Default ``"sstp"``.

    Example::

        agent = LLMAgent(
            "mediator",
            prefer_low=True,
            persona=(
                "You are a neutral mediator.  Prefer socially optimal outcomes. "
                "Accept any offer with utility above 0.4."
            ),
            prompt_mode="english",
        )
    """

    def __init__(
        self,
        agent_id: str,
        prefer_low: bool = True,
        persona: str = "You are a negotiation agent. Act rationally.",
        prompt_mode: str = "sstp",
    ) -> None:
        super().__init__(agent_id, prefer_low)
        self.persona = persona
        self.prompt_mode = prompt_mode
        self._llm: Optional[Callable[[str], str]] = None  # lazy init

    def _ensure_llm(self) -> Callable[[str], str]:
        if self._llm is None:
            self._llm = _get_llm()
        return self._llm

    # ------------------------------------------------------------------
    # JSON extraction (robust to prose + markdown fences)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Return the first balanced JSON object found in *text*."""
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-z]*\n?", "", stripped).rstrip("`").strip()
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object in LLM response: {text!r}")
        depth, in_str, escape_next = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_str:
                escape_next = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
        raise ValueError(f"Unbalanced braces in LLM response: {text!r}")

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _utility_table(self, options_per_issue: Dict[str, List[str]]) -> str:
        prefs = self._build_prefs(options_per_issue)
        lines: List[str] = []
        for issue, opts in options_per_issue.items():
            for opt in opts:
                u = prefs[issue].get(opt, 0.0)
                lines.append(f"  [{issue}] '{opt}' → {u:.3f}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        body: Dict[str, Any],
        action: str,
        round_num: int,
        n_steps: int,
        issues: List[str],
        options_per_issue: Dict[str, List[str]],
        current_offer: Dict[str, str],
        sao_state_raw: Dict[str, Any],
        sc: Dict[str, Any],
    ) -> str:
        return (
            f"{self.persona}\n\n"
            f"```json\n{json.dumps(body, indent=2)}\n```\n\n"
            f"Reply ONLY with JSON (no markdown): "
            f'{{"response": 0|1, "outcome": <offer dict or null>, '
            f'"reason": "<1-3 sentences explaining your decision>"}}\n'
            f"ResponseType: 0 = ACCEPT_OFFER, 1 = REJECT_OFFER. "
            f"If counter-offering, set response=1 and outcome to your proposed offer."
        )

    # ------------------------------------------------------------------
    # Fallback decisions (used when LLM call fails)
    # ------------------------------------------------------------------

    def _fallback_propose(
        self, options_per_issue: Dict[str, List[str]]
    ) -> Dict[str, str]:
        """Return the ideal offer (best outcome for this agent's preference)."""
        return {
            issue: (opts[0] if self.prefer_low else opts[-1])
            for issue, opts in options_per_issue.items()
        }

    # ------------------------------------------------------------------
    # Override: handle_message reads the full SSTP envelope
    # ------------------------------------------------------------------

    def handle_message(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Read the full SSTPNegotiateMessage and ask the LLM to decide.

        Overrides :meth:`BaseAgent.handle_message` to pass the raw envelope
        to the LLM prompt rather than just structured fields.
        """
        payload: Dict[str, Any] = body.get("payload") or {}
        sc: Dict[str, Any] = body.get("semantic_context") or {}

        action: str = payload.get("action", "respond")
        round_num: int = payload.get("round", 1)
        n_steps: int = payload.get("n_steps") or 200
        current_offer: Dict[str, str] = payload.get("current_offer") or {}
        issues: List[str] = sc.get("issues") or []
        options_per_issue: Dict[str, List[str]] = sc.get("options_per_issue") or {}
        session_id: str = sc.get("session_id") or "unknown"
        sao_state_raw: Dict[str, Any] = sc.get("sao_state") or {}

        from protocol.sstp.negmas_sao import SAOState  # noqa

        incoming_sao_state: Optional[SAOState] = (
            SAOState(**sao_state_raw) if sao_state_raw else None
        )

        # Skip LLM for empty respond calls (no offer on table)
        if action == "respond" and not current_offer:
            reply_payload = attach_reason(
                {
                    "action": "reject",
                    "round": round_num,
                    "issues": issues,
                    "options_per_issue": options_per_issue,
                },
                "No current offer on the table.",
            )
            sao_resp = SAOResponse(response=ResponseType.REJECT_OFFER, outcome=None)
            return build_sstp_reply(
                session_id,
                self.agent_id,
                reply_payload,
                sao_response=sao_resp,
                sao_state=incoming_sao_state,
            )

        prompt = self._build_prompt(
            body,
            action,
            round_num,
            n_steps,
            issues,
            options_per_issue,
            current_offer,
            sao_state_raw,
            sc,
        )

        action_str, outcome, sao_resp, reason = self._call_llm(
            prompt, action, round_num, current_offer, options_per_issue
        )

        if action_str == "counter_offer":
            reply_payload = attach_reason(
                {
                    "action": "counter_offer",
                    "round": round_num,
                    "issues": issues,
                    "options_per_issue": options_per_issue,
                    "offer": outcome,
                },
                reason,
            )
        elif action_str == "accept":
            reply_payload = attach_reason(
                {
                    "action": "accept",
                    "round": round_num,
                    "issues": issues,
                    "options_per_issue": options_per_issue,
                },
                reason,
            )
        else:
            reply_payload = attach_reason(
                {
                    "action": "reject",
                    "round": round_num,
                    "issues": issues,
                    "options_per_issue": options_per_issue,
                },
                reason,
            )

        print(
            f"  [{self.agent_id}] {action}  round={round_num}"
            f"  → {action_str}" + (f"  offer={outcome}" if outcome else "")
            + f"  reason={reason[:80]!r}",
            flush=True,
        )

        return build_sstp_reply(
            session_id,
            self.agent_id,
            reply_payload,
            sao_response=sao_resp,
            sao_state=incoming_sao_state,
        )

    def _call_llm(
        self,
        prompt: str,
        action: str,
        round_num: int,
        current_offer: Dict[str, str],
        options_per_issue: Dict[str, List[str]],
    ) -> Tuple[str, Optional[Dict[str, str]], SAOResponse, str]:
        """Call the LLM and parse its response.  Falls back on any error.

        Returns:
            ``(action_str, outcome_or_None, sao_response, reason)``
            where ``action_str`` is ``"counter_offer"`` | ``"accept"`` | ``"reject"``.
        """
        try:
            llm = self._ensure_llm()
            raw = llm(prompt)
            data = self._extract_json(raw)
            resp_int = int(data.get("response", 1))
            outcome_raw = data.get("outcome")
            # Required in prompts; fallback to _default_reason if missing/invalid.
            reason_raw = sanitize_reason(data.get("reason"))

            if resp_int == int(ResponseType.ACCEPT_OFFER):
                outcome = current_offer or (
                    outcome_raw if isinstance(outcome_raw, dict) else {}
                )
                reason = reason_raw or self._default_reason(
                    "respond", decision="accept"
                )
                return (
                    "accept",
                    outcome,
                    SAOResponse(response=ResponseType.ACCEPT_OFFER, outcome=outcome),
                    reason,
                )
            else:  # REJECT_OFFER
                if outcome_raw and isinstance(outcome_raw, dict) and options_per_issue:
                    # Validate — snap unknown options to the agent's ideal
                    validated = {
                        issue: (
                            outcome_raw.get(issue)
                            if outcome_raw.get(issue) in opts
                            else (opts[0] if self.prefer_low else opts[-1])
                        )
                        for issue, opts in options_per_issue.items()
                    }
                    reason = reason_raw or self._default_reason(
                        "propose", decision="counter_offer"
                    )
                    return (
                        "counter_offer",
                        validated,
                        SAOResponse(
                            response=ResponseType.REJECT_OFFER, outcome=validated
                        ),
                        reason,
                    )
                reason = reason_raw or self._default_reason(
                    "respond", decision="reject"
                )
                return (
                    "reject",
                    None,
                    SAOResponse(response=ResponseType.REJECT_OFFER, outcome=None),
                    reason,
                )

        except Exception as exc:  # noqa: BLE001
            print(f"  [{self.agent_id}] LLM error (fallback): {exc}", flush=True)
            if action == "propose":
                fallback = self._fallback_propose(options_per_issue)
                return (
                    "counter_offer",
                    fallback,
                    SAOResponse(response=ResponseType.REJECT_OFFER, outcome=fallback),
                    self._default_reason("propose", decision="counter_offer"),
                )
            return (
                "reject",
                None,
                SAOResponse(response=ResponseType.REJECT_OFFER, outcome=None),
                self._default_reason("respond", decision="reject"),
            )

    # ------------------------------------------------------------------
    # BaseAgent abstract methods — shims when handle_message is bypassed
    # ------------------------------------------------------------------

    def decide_propose(
        self,
        round_num: int,
        n_steps: int,
        options_per_issue: Dict[str, List[str]],
    ) -> Tuple[Dict[str, str], None]:
        """Shim: builds a minimal body and delegates to handle_message."""
        body: Dict[str, Any] = {
            "payload": {"action": "propose", "round": round_num, "n_steps": n_steps},
            "semantic_context": {
                "issues": list(options_per_issue.keys()),
                "options_per_issue": options_per_issue,
                "sao_state": {"step": round_num, "n_steps": n_steps},
                "session_id": "shim",
            },
        }
        reply = self.handle_message(body)
        offer = (reply.get("payload") or {}).get("offer") or self._fallback_propose(
            options_per_issue
        )
        return offer, None

    def decide_respond(
        self,
        offer: Dict[str, str],
        round_num: int,
        n_steps: int,
        options_per_issue: Dict[str, List[str]],
    ) -> str:
        """Shim: builds a minimal body and delegates to handle_message."""
        body: Dict[str, Any] = {
            "payload": {
                "action": "respond",
                "round": round_num,
                "n_steps": n_steps,
                "current_offer": offer,
            },
            "semantic_context": {
                "issues": list(options_per_issue.keys()),
                "options_per_issue": options_per_issue,
                "sao_state": {
                    "step": round_num,
                    "n_steps": n_steps,
                    "current_offer": offer,
                },
                "session_id": "shim",
            },
        }
        reply = self.handle_message(body)
        action = (reply.get("payload") or {}).get("action", "reject")
        return "accept" if action == "accept" else "reject"
