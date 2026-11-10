"""ABCD JSON -> customer-side SFT JSONL.

We flip the usual chat roles: the model plays the CUSTOMER.

    system    = the ABCD scenario (who I am, what I want)
    user      = support agent turns
    assistant = customer turns       <- the only turns we train on

ABCD "action" turns (agent tool calls such as "Account has been pulled up
for ...") are dropped or turned into a short note inside the agent turn,
depending on ``action_mode``. The final customer message ends with
``<END_CONVERSATION>`` so the simulator learns when to stop.

Splits are made by GROUP (intent/subflow by default) so that no scenario
template is seen in both train and test (risk R4).

Usage:
    python -m usersim.prepare_data --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

END_TOKEN = "<END_CONVERSATION>"
CHAT_START = "[customer joined the chat]"  # placeholder when the customer speaks first
ROLE_MAP = {"agent": "user", "customer": "assistant"}

# ---------------------------------------------------------------------------
# PII masking (R6). ABCD identities are synthetic, so masking is off by default
# for ABCD; turn it on for any real-customer data (e.g. Twitter support logs).
# ---------------------------------------------------------------------------

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"(?:\+?\d{1,2}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"), "[PHONE]"),  # needs separators
    (re.compile(r"(?<!\w)@\w{2,}"), "[HANDLE]"),
    (re.compile(r"\b\d{5,}\b"), "[NUMBER]"),
]


def mask_pii(text: str) -> str:
    """Mask emails, phone numbers, @handles and long digit runs (order of rules matters)."""
    for pattern, token in PII_PATTERNS:
        text = pattern.sub(token, text)
    return text


# ---------------------------------------------------------------------------
# Scenario -> system prompt (also used by the simulators at inference time)
# ---------------------------------------------------------------------------


def render_scenario(scenario: dict) -> str:
    """Turn an ABCD scenario dict into the system prompt the customer model sees."""
    personal = scenario.get("personal", {})
    order = scenario.get("order", {})
    products = ", ".join(n.replace("_", " ") for n in scenario.get("product", {}).get("names", []))
    lines = [
        "You are a customer chatting with an online clothing store's support agent.",
        f"Your issue: {scenario.get('flow', '?').replace('_', ' ')} "
        f"({scenario.get('subflow', '?').replace('_', ' ')}).",
        "What you know about yourself:",
    ]
    for key in ("customer_name", "username", "email", "phone", "member_level"):
        if personal.get(key):
            lines.append(f"- {key.replace('_', ' ')}: {personal[key]}")
    for key in ("order_id", "purchase_date", "payment_method", "full_address"):
        if order.get(key):
            lines.append(f"- {key.replace('_', ' ')}: {order[key]}")
    if products:
        lines.append(f"- products: {products}")
    lines.append(f"When the conversation is over for you, end your message with {END_TOKEN}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversation -> chat messages
# ---------------------------------------------------------------------------


def to_turns(convo: dict, action_mode: str = "drop") -> list[dict]:
    """ABCD ``original`` -> [{"role": "customer"|"agent", "text": ...}].

    action_mode: "drop" removes action turns, "note" keeps them as
    "[action] ..." text inside the agent's side of the chat.
    """
    turns = []
    for speaker, text in convo["original"]:
        if speaker == "action":
            if action_mode == "drop":
                continue
            speaker, text = "agent", f"[action] {text}"
        turns.append({"role": speaker, "text": text.strip()})
    return turns


def to_messages(convo: dict, action_mode: str = "drop", pii: bool = False) -> list[dict]:
    """Build the SFT chat: system (scenario) + alternating user/assistant messages."""
    messages = []
    for turn in to_turns(convo, action_mode):
        role = ROLE_MAP[turn["role"]]
        text = mask_pii(turn["text"]) if pii else turn["text"]
        if messages and messages[-1]["role"] == role:  # merge consecutive same-speaker turns
            messages[-1]["content"] += "\n" + text
        else:
            messages.append({"role": role, "content": text})
    if not messages or messages[0]["role"] == "assistant":
        messages.insert(0, {"role": "user", "content": CHAT_START})
    if messages[-1]["role"] == "assistant":
        messages[-1]["content"] += "\n" + END_TOKEN
    else:  # the agent spoke last: the customer's "reply" is to leave
        messages.append({"role": "assistant", "content": END_TOKEN})
    system = render_scenario(convo["scenario"])
    return [{"role": "system", "content": mask_pii(system) if pii else system}] + messages


# ---------------------------------------------------------------------------
# Grouped split (R4)
# ---------------------------------------------------------------------------


def group_key(convo: dict, group_by: str) -> str:
    scenario = convo["scenario"]
    if group_by == "subflow":
        return scenario["subflow"]
    if group_by == "flow":
        return scenario["flow"]
    if group_by == "scenario":  # same customer + same order = same scenario
        key = json.dumps([scenario.get("personal"), scenario.get("order")], sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()
    if group_by == "convo":
        return str(convo["convo_id"])
    raise ValueError(f"unknown group_by: {group_by}")


def group_split(convos: list[dict], group_by: str, dev: float, test: float, seed: int) -> dict:
    """Assign whole groups to test, then dev, then train, until each ratio is reached."""
    sizes = Counter(group_key(c, group_by) for c in convos)
    groups = sorted(sizes)
    random.Random(seed).shuffle(groups)
    targets = {"test": test * len(convos), "dev": dev * len(convos)}
    assignment, filled = {}, {"test": 0, "dev": 0}
    for g in groups:
        split = next((s for s in ("test", "dev") if filled[s] < targets[s]), "train")
        assignment[g] = split
        if split != "train":
            filled[split] += sizes[g]
    out = {"train": [], "dev": [], "test": []}
    for c in convos:
        out[assignment[group_key(c, group_by)]].append(c)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_abcd(path: str | Path) -> list[dict]:
    """Pool ABCD's own train/dev/test; we re-split by group ourselves."""
    data = json.loads(Path(path).read_text())
    return [c for split in ("train", "dev", "test") for c in data.get(split, [])]


def make_record(convo: dict, cfg: dict) -> dict:
    """One JSONL line: SFT messages + plain turns/scenario for evaluation."""
    return {
        "convo_id": convo["convo_id"],
        "group": group_key(convo, cfg["group_by"]),
        "scenario": convo["scenario"],
        "turns": to_turns(convo, "drop"),
        "messages": to_messages(convo, cfg["action_mode"], cfg["mask_pii"]),
    }


def prepare(cfg: dict) -> dict:
    convos = load_abcd(cfg["abcd_path"])
    if cfg.get("max_conversations"):
        convos = convos[: cfg["max_conversations"]]
    splits = group_split(convos, cfg["group_by"], cfg["dev_ratio"], cfg["test_ratio"], cfg["seed"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, items in splits.items():
        records = [make_record(c, cfg) for c in items]
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        stats[name] = {"conversations": len(records), "groups": len({r["group"] for r in records})}
    groups = {n: {group_key(c, cfg["group_by"]) for c in items} for n, items in splits.items()}
    assert not groups["train"] & groups["test"], "leak: group in both train and test"
    assert not groups["train"] & groups["dev"], "leak: group in both train and dev"
    stats["test_groups"] = sorted(groups["test"])
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    import yaml

    cfg = yaml.safe_load(open(args.config))
    print(json.dumps(prepare(cfg), indent=2))


if __name__ == "__main__":
    main()
