import json

import pytest

from conftest import fake_convo
from usersim import prepare_data as P


def test_role_mapping_and_end_token():
    msgs = P.to_messages(fake_convo(1), action_mode="drop")
    assert msgs[0]["role"] == "system" and "return size" in msgs[0]["content"]
    # agent -> user, customer -> assistant; consecutive agent turns merged
    assert msgs[1] == {"role": "user", "content": "Hi!\nHow can I help you?"}
    assert msgs[2] == {"role": "assistant", "content": "I need to return my jeans, they don't fit."}
    roles = [m["role"] for m in msgs[1:]]
    assert all(a != b for a, b in zip(roles, roles[1:])), "roles must alternate"
    # the agent spoke last, so the customer's final move is to end the chat
    assert msgs[-1] == {"role": "assistant", "content": P.END_TOKEN}
    assert not any("[action]" in m["content"] or "pulled up" in m["content"] for m in msgs)


def test_action_mode_note_keeps_action_on_agent_side():
    msgs = P.to_messages(fake_convo(1), action_mode="note")
    action_msgs = [m for m in msgs if "[action]" in m["content"]]
    assert len(action_msgs) == 1 and action_msgs[0]["role"] == "user"


def test_customer_first_gets_placeholder_and_end_appended():
    c = fake_convo(2)
    c["original"] = [["customer", "hi, I want a refund"], ["agent", "sure"], ["customer", "thanks bye"]]
    msgs = P.to_messages(c)
    assert msgs[1] == {"role": "user", "content": P.CHAT_START}
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["content"].endswith(P.END_TOKEN)
    assert msgs[-1]["content"].startswith("thanks bye")


def test_mask_pii():
    text = "mail me at jo.doe@mail.com or call (555) 123-4567, @jodoe, order 3348917502"
    masked = P.mask_pii(text)
    assert masked == "mail me at [EMAIL] or call [PHONE], [HANDLE], order [NUMBER]"
    assert P.mask_pii("I have 2 shirts") == "I have 2 shirts"


@pytest.mark.parametrize("group_by", ["subflow", "flow", "scenario", "convo"])
def test_group_split_has_no_leakage(abcd_like, group_by):
    convos = abcd_like["train"] + abcd_like["dev"] + abcd_like["test"]
    splits = P.group_split(convos, group_by, dev=0.15, test=0.2, seed=0)
    groups = {s: {P.group_key(c, group_by) for c in items} for s, items in splits.items()}
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["dev"]
    assert not groups["dev"] & groups["test"]
    assert sum(len(v) for v in splits.values()) == len(convos)
    assert splits["test"] and splits["train"]


def test_prepare_end_to_end(tmp_path, abcd_like):
    raw = tmp_path / "abcd.json"
    raw.write_text(json.dumps(abcd_like))
    cfg = {"abcd_path": raw, "out_dir": tmp_path / "out", "action_mode": "drop", "mask_pii": False,
           "group_by": "subflow", "dev_ratio": 0.15, "test_ratio": 0.2, "seed": 0}
    stats = P.prepare(cfg)
    assert sum(stats[s]["conversations"] for s in ("train", "dev", "test")) == 60
    train = [json.loads(l) for l in open(tmp_path / "out" / "train.jsonl")]
    test = [json.loads(l) for l in open(tmp_path / "out" / "test.jsonl")]
    test_subflows = {r["scenario"]["subflow"] for r in test}
    assert not test_subflows & {r["scenario"]["subflow"] for r in train}  # unseen intents in test
    record = train[0]
    assert {"messages", "turns", "scenario", "group"} <= set(record)
    assert all(t["role"] in {"customer", "agent"} for t in record["turns"])
