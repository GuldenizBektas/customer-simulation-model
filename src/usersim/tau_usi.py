# =============================================================================
#  EVALUATION ONLY — license forbids training
#
#  cmu-lti/tau-usi (CC BY-NC 4.0 + extra terms) prohibits using the data to
#  train, fine-tune, distill or otherwise improve any LLM, and forbids
#  commercial use. In this project it is ONLY used as out-of-domain human
#  reference data for metrics (H2). Never feed it to prepare_data/train, never
#  put it in prompts, never redistribute it. Cite arXiv:2603.11245.
# =============================================================================
"""Load tau-usi human conversations into the metrics format.

Known top-level fields (dataset card): task_id, instance_id ("airline_0",
"retail_42_ann2"), agent_id, conversation, survey, quality_score,
quality_passed, keep, reward (0/1).

The inner structure of ``conversation`` is not documented on the card (the
dataset is gated). The role/text keys below are guesses -> TODO verify after
accepting the terms and downloading data/tau_bench_tasks_unified.json.
"""

from __future__ import annotations

import json
from pathlib import Path

# TODO(verify): actual keys inside each conversation message.
ROLE_KEY = "role"
TEXT_KEY = "content"
CUSTOMER_ROLES = {"user", "human", "customer"}  # TODO(verify)
AGENT_ROLES = {"assistant", "agent"}  # tool messages and anything else are skipped


def convert(entry: dict) -> dict | None:
    turns = []
    for msg in entry.get("conversation", []):
        role, text = msg.get(ROLE_KEY), msg.get(TEXT_KEY)
        if not text:
            continue
        if role in CUSTOMER_ROLES:
            turns.append({"role": "customer", "text": text})
        elif role in AGENT_ROLES:
            turns.append({"role": "agent", "text": text})
    if not turns:
        return None
    return {
        "turns": turns,
        "scenario": None,  # tau-bench tasks are not ABCD scenarios; goal metrics -> NaN
        "end_reason": None,
        "meta": {
            "instance_id": entry.get("instance_id"),
            "domain": str(entry.get("instance_id", "")).split("_")[0],  # airline / retail
            "agent_id": entry.get("agent_id"),
            "reward": entry.get("reward"),  # human-run agent success (H3 reference)
        },
    }


def load_tau_usi(path: str | Path, keep_only: bool = True) -> list[dict]:
    data = json.loads(Path(path).read_text())
    entries = data if isinstance(data, list) else list(data.values())  # TODO(verify) top-level shape
    if keep_only:
        entries = [e for e in entries if e.get("keep", True)]
    return [c for c in (convert(e) for e in entries) if c]


def human_success_rate(convos: list[dict]) -> float:
    """Mean tau-bench reward over human conversations (paper reports 63.6%)."""
    rewards = [c["meta"]["reward"] for c in convos if c["meta"]["reward"] is not None]
    return sum(rewards) / len(rewards) if rewards else float("nan")
