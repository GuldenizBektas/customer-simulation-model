"""Fake ABCD-shaped data for tests (no real data is needed or downloaded)."""

import pytest

FLOWS = [("product_defect", "return_size"), ("product_defect", "refund_initiate"),
         ("account_access", "recover_password"), ("shipping_issue", "status_delivery"),
         ("order_issue", "cancel_order"), ("subscription_inquiry", "status_due_date")]


def fake_scenario(i: int, flow: str, subflow: str) -> dict:
    return {
        "personal": {"customer_name": f"alex smith{i}", "email": f"alex{i}@email.com",
                     "member_level": "gold", "phone": "(555) 010-0199", "username": f"asmith{i}0"},
        "order": {"order_id": f"33489175{i:02d}", "full_address": "12 1st ave springfield ny 75227",
                  "payment_method": "credit card", "purchase_date": "2019-11-06"},
        "product": {"names": ["michael_kors jeans"], "amounts": [94]},
        "flow": flow,
        "subflow": subflow,
    }


def fake_convo(i: int, flow: str = "product_defect", subflow: str = "return_size") -> dict:
    sc = fake_scenario(i, flow, subflow)
    original = [
        ["agent", "Hi!"],
        ["agent", "How can I help you?"],
        ["customer", "I need to return my jeans, they don't fit."],
        ["agent", "Sure. May I have your full name?"],
        ["customer", f"alex smith{i}"],
        ["action", f"Account has been pulled up for alex smith{i}."],
        ["agent", "What is the order id?"],
        ["customer", sc["order"]["order_id"]],
        ["agent", "Done, the return is set up."],
        ["customer", "thanks!"],
        ["agent", "Have a nice day."],
    ]
    return {"convo_id": 1000 + i, "scenario": sc, "original": original, "delexed": []}


@pytest.fixture
def abcd_like():
    """{train, dev, test} dict like abcd_v1.1.json, 60 conversations over 6 subflows."""
    convos = [fake_convo(i, *FLOWS[i % len(FLOWS)]) for i in range(60)]
    return {"train": convos[:40], "dev": convos[40:50], "test": convos[50:]}


def convo(*customer_turns, agent="ok.", scenario=None, end_reason=None):
    """Build a metrics-format conversation with an agent turn before each customer turn."""
    turns = []
    for text in customer_turns:
        turns += [{"role": "agent", "text": agent}, {"role": "customer", "text": text}]
    return {"turns": turns, "scenario": scenario, "end_reason": end_reason}
