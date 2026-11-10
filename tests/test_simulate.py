"""The conversation loop and evaluation helpers, run without any LLM."""

import numpy as np

from conftest import fake_scenario
from usersim import evaluate as E
from usersim.agents import RuleBasedAgent, split_resolution
from usersim.simulate import run_conversation
from usersim.simulators import scripted


def test_split_resolution():
    assert split_resolution("Done! [RESOLVED: return_size] Bye") == ("Done!  Bye", "return_size")
    assert split_resolution("hello") == ("hello", None)


def test_rule_agent_resolves_and_customer_ends():
    sc = fake_scenario(1, "product_defect", "return_size")
    sim = scripted(["I want to return jeans", "alex smith1", "3348917501", "thanks! <END_CONVERSATION>"])
    turns, end = run_conversation(sim, RuleBasedAgent(), sc, max_turns=10)
    assert end == "customer_end"
    assert turns[-1] == {"role": "customer", "text": "thanks!"}
    record = {"turns": turns, "scenario": sc, "end_reason": end}
    assert E.agent_success(record) == 1.0


def test_looping_rule_agent_hits_max_turns():
    sc = fake_scenario(1, "product_defect", "return_size")
    sim = scripted(["return please", "alex smith1", "3348917501"])
    turns, end = run_conversation(sim, RuleBasedAgent(fault="loop"), sc, max_turns=6)
    assert end == "max_turns"
    assert E.agent_success({"turns": turns, "scenario": sc, "end_reason": end}) == 0.0


def test_cohens_d_and_bootstrap():
    assert E.cohens_d([0, 1, 0, 1], [0, 1, 0, 1]) == 0.0
    assert E.cohens_d([0, 0, 1, 1], [1, 1, 2, 2]) > 1
    lo, hi = E.bootstrap_ci(np.mean, np.arange(100), n=200, seed=0)
    assert lo < 49.5 < hi


def test_ngram_overlap():
    train = E.ngrams("where is my order please", 3)
    c = {"turns": [{"role": "customer", "text": "where is my order now"}]}
    assert E.ngram_overlap([c], train, 3) == 2 / 3


def test_cohens_d_zero_variance():
    assert E.cohens_d([1, 1, 1], [0, 0, 0]) == float("-inf")
