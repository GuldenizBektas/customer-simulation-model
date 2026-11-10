import math

import numpy as np
import pytest

from conftest import convo, fake_scenario
from usersim import metrics as M


def test_words_per_turn():
    assert M.words_per_turn(convo("one two", "one two three four")) == 3.0
    assert math.isnan(M.words_per_turn(convo()))


def test_short_turn_rate():
    assert M.short_turn_rate(convo("yes", "no thanks", "this one is long enough")) == pytest.approx(2 / 3)


def test_polite_rate():
    assert M.polite_rate(convo("please help", "Thank you!", "my jeans ripped", "sorry")) == 0.75


def test_dash_rate():
    assert M.dash_rate(convo("I need help — urgently", "plain-hyphen is fine")) == 0.5


def test_ack_rate_only_counts_pure_acks():
    c = convo("ok", "Got it.", "sure, thanks", "ok my order id is 123", "Okay okay!")
    assert M.ack_rate(c) == pytest.approx(3 / 5)  # "sure, thanks" and the id turn are not acks


def test_verbosity_cv():
    assert M.verbosity_cv(convo("a b", "a b")) == 0.0
    assert M.verbosity_cv(convo("a", "a b c")) == pytest.approx(0.5)
    assert math.isnan(M.verbosity_cv(convo("only one turn")))


def test_repeat_flag():
    assert M.repeat_flag(convo(*["where is my order"] * 6)) == 1.0
    assert M.repeat_flag(convo(*["where is my order"] * 5)) == 0.0


def test_identity_confusion():
    assert M.identity_confusion(convo("Hello, how may I help you today?")) == 1.0
    assert M.identity_confusion(convo("my order is late")) == 0.0


def test_front_loading():
    assert M.front_loading(convo("a b c", "d", "e f g h", "i j")) == pytest.approx(4 / 10)


def test_ids_per_turn():
    c = convo("my order is 3348917502", "email alex1@email.com, user asmith10", "no id here")
    assert M.ids_per_turn(c) == pytest.approx((1 + 2 + 0) / 3)


def test_opening_words():
    assert M.opening_words(convo("hi my jeans are too big", "ok")) == 6.0


def test_classify_question_priority():
    assert M.classify_question("Are you sure? what do you mean?") == "pushback"
    assert M.classify_question("what do you mean?") == "clarify"
    assert M.classify_question("can you check the status?") == "info"
    assert M.classify_question("my name is alex") is None


def test_uncertainty_and_certainty_rate():
    c = convo("I think it was last week", "definitely the blue one", "ok")
    assert M.uncertainty_rate(c) == pytest.approx(1 / 3)
    assert M.certainty_rate(c) == pytest.approx(1 / 3)


def test_question_rates():
    c = convo("you already asked that", "what do you mean by code?", "where is my package?", "fine")
    assert M.pushback_rate(c) == 0.25
    assert M.clarify_q_rate(c) == 0.25
    assert M.info_q_rate(c) == 0.25


def test_emotion_accuse_pivot_rates():
    c = convo("ugh this is so frustrating", "you are useless", "never mind, let's try a refund instead", "ok")
    assert M.emotion_rate(c) == 0.25
    assert M.accuse_rate(c) == 0.25
    assert M.pivot_rate(c) == 0.25


def test_resolved_closure_and_abandoned():
    happy = convo("my order is late", "great, thanks that solved it")
    angry = convo("my order is late", "this is useless, bye")
    assert M.resolved_closure(happy) == 1.0 and M.abandoned(happy) == 0.0
    assert M.resolved_closure(angry) == 0.0 and M.abandoned(angry) == 1.0
    cut = convo("my order is late", "still waiting", end_reason="max_turns")
    assert M.abandoned(cut) == 0.0  # hitting max_turns is not the customer leaving


def test_goal_adherence():
    sc = fake_scenario(1, "product_defect", "return_size")
    full = convo("return my jeans please", "alex smith1", "order 3348917501", "user asmith10, alex1@email.com", scenario=sc)
    partial = convo("return my jeans please", "alex smith1", scenario=sc)
    assert M.goal_adherence(full) == 1.0
    assert M.goal_adherence(partial) == pytest.approx(2 / 5)  # name + product of 5 slots
    assert math.isnan(M.goal_adherence(convo("hi")))  # no scenario


def test_offscript_id_rate():
    sc = fake_scenario(1, "product_defect", "return_size")
    c = convo("order 3348917501", "actually it is order 9999999999", "thanks", scenario=sc)
    assert M.offscript_id_rate(c) == pytest.approx(1 / 3)


def test_dice_similarity():
    assert M.dice_similarity(10, 10) == 100.0
    assert M.dice_similarity(0, 0) == 100.0
    assert M.dice_similarity(5, 15) == pytest.approx(50.0)
    assert M.dice_similarity(0, 4) == 0.0


def test_wasserstein_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    u, v = rng.normal(0, 1, 50), rng.normal(1, 2, 70)
    assert M.wasserstein_1d(u, v) == pytest.approx(stats.wasserstein_distance(u, v))
    assert M.wasserstein_1d([1, 2, 3], [1, 2, 3]) == 0.0


def test_compare_identical_corpora_scores_100():
    corpus = [convo("hi, my jeans ripped", "ok", "thanks!"), convo("where is my order?", "ugh", "fine")]
    result = M.compare(corpus, corpus)
    assert result["composite"] == pytest.approx(100.0)
    assert set(result["dimensions"]) == {"D1", "D2", "D3", "D4"}
    assert all(r["w1"] == 0.0 for r in result["metrics"].values() if not math.isnan(r["w1"]))


def test_compare_detects_different_style():
    human = [convo("ugh where is my order", "no", "you already asked!!")] * 5
    llm = [convo("Hello! I would be very grateful if you could kindly help me locate my order today.",
                 "Thank you so much for your wonderful assistance — I truly appreciate it.")] * 5
    result = M.compare(llm, human)
    # Many rare markers are 0 in both corpora and score 100 (paper's Dice rule),
    # so the composite stays well above 0 even for very different styles.
    assert result["composite"] < 75
    assert result["metrics"]["wds_per_turn"]["dice"] < 50


def test_human_ceiling_is_high_for_homogeneous_data():
    corpus = [convo("my order is late", "ok", "thanks")] * 10 + [convo("ugh", "useless")] * 10
    ceiling = M.human_ceiling(corpus, n_splits=5, seed=0)
    assert 0 <= ceiling["mean"] <= 100 and len(ceiling["scores"]) == 5


def test_every_metric_is_registered_and_returns_float():
    sc = fake_scenario(1, "product_defect", "return_size")
    c = convo("hi", "my order 3348917501", "thanks", scenario=sc)
    for name, (dim, fn) in M.METRICS.items():
        assert dim in {"D1", "D2", "D3", "D4", "X"}
        assert isinstance(fn(c), float), name
