"""Behavioral metrics for customer simulators.

Re-implementation of the D1-D4 sub-metrics from Zhou et al., "Mind the Sim2Real
Gap in User Simulation for Agentic Tasks" (arXiv:2603.11245, Appendix A),
plus a few extras (abandonment, resolved closure, goal adherence).

Conventions
-----------
A *conversation* is a dict:

    {"turns": [{"role": "customer" | "agent", "text": str}, ...],
     "scenario": dict | None,      # ABCD scenario (needed for goal metrics)
     "end_reason": str | None}     # "customer_end", "max_turns", ... (sims only)

Every metric is a function ``metric(convo) -> float`` computed on ONE
conversation (NaN if undefined). A corpus value is the mean over conversations,
and the list of per-conversation values is the distribution we compare
(Wasserstein). Note: the paper pools turns across the corpus for turn-level
rates; averaging per conversation weights each conversation equally instead.

All lexicons live in the "Lexicons" block below so they can be audited (and
criticised) in one place. They are illustrative, not exhaustive; the paper
gives examples, not full lists.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Lexicons and regexes (single source of truth)
# ---------------------------------------------------------------------------


def _phrases(*patterns: str) -> re.Pattern:
    """Compile regex fragments into one case-insensitive, word-bounded pattern."""
    return re.compile(r"(?<!\w)(?:" + "|".join(patterns) + r")(?!\w)", re.IGNORECASE)


POLITE = _phrases(r"please", r"pls", r"thanks?", r"thank you", r"thx", r"sorry", r"appreciate\w*")
DASH = re.compile("[–—]")  # en dash, em dash
_ACK = r"(?:ok(?:ay)?|k|sure|got it|alright|all right|cool|sounds good|understood|noted)"
ACK_ONLY = re.compile(rf"^\s*{_ACK}(?:[\s,.!]+{_ACK})*[\s.!]*$", re.IGNORECASE)
AGENT_LANGUAGE = _phrases(
    r"how (?:may|can) i (?:help|assist)",
    r"let me check",
    r"for verification purposes",
    r"is there anything else i can (?:help|do)",
    r"i(?:'ve| have) pulled up your account",
    r"thank you for contacting",
)
# Identifier-like tokens: emails, long digit runs, alphanumerics with a digit.
ID_TOKEN = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+"  # email
    r"|\b(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{6,}\b"  # >=6 chars, contains a digit
    r"|\b\d{5}\b"  # 5-digit runs (zip codes, short ids)
)
UNCERTAIN = _phrases(
    r"maybe", r"perhaps", r"not sure", r"i think", r"i guess", r"probably",
    r"might", r"possibly", r"i don'?t know", r"idk", r"not certain",
)
CERTAIN = _phrases(
    r"definitely", r"for sure", r"without a doubt", r"certainly", r"absolutely",
    r"i'?m sure", r"i am sure", r"positive",
)
PUSHBACK = _phrases(
    r"are you sure", r"you already asked", r"i already (?:told|gave|said|provided|sent)",
    r"i just (?:told|gave|said)", r"that'?s not what i", r"that is not what i",
    r"that'?s (?:wrong|not right|incorrect)", r"why do you need", r"you said",
)
CLARIFY = _phrases(
    r"what do you mean", r"can you clarify", r"could you clarify", r"what does that mean",
    r"i don'?t understand", r"do you mean", r"which one", r"meaning\?",
)
INFO_Q_START = re.compile(
    r"^\s*(?:what|when|where|why|how|which|who|can you|could you|will|would|is|are|do|does|did)\b",
    re.IGNORECASE,
)
EMOTION = _phrases(
    r"frustrat\w*", r"annoy\w*", r"ugh+", r"angry", r"upset", r"disappoint\w*",
    r"fed up", r"sick of", r"tired of", r"seriously\?*", r"wtf", r"omg", r"!{2,}",
)
ACCUSE = _phrases(
    r"useless", r"unacceptable", r"scam", r"incompetent", r"terrible", r"worst",
    r"ridiculous", r"waste of (?:my )?time", r"you(?:'re| are) not (?:listening|helping)",
    r"rip ?off", r"pathetic",
)
PIVOT = _phrases(
    r"instead", r"on second thought", r"let'?s try", r"let me try", r"never ?mind",
    r"forget (?:it|that)", r"how about", r"what if", r"alternatively",
)
RESOLVED_CLOSURE = _phrases(
    r"thanks?", r"thank you", r"appreciate\w*", r"that'?s all", r"that is all", r"all set",
    r"perfect", r"great", r"awesome",
    r"that (?:helps|helped|works|worked|solved|fixes|fixed)", r"problem solved",
)
END_TOKEN = "<END_CONVERSATION>"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def customer_texts(convo: dict) -> list[str]:
    return [t["text"] for t in convo["turns"] if t["role"] == "customer"]


def n_words(text: str) -> int:
    return len(text.split())


def _rate(texts: list[str], predicate) -> float:
    """Share of turns for which ``predicate(text)`` is true (NaN if no turns)."""
    if not texts:
        return float("nan")
    return float(np.mean([bool(predicate(t)) for t in texts]))


def _matches(pattern: re.Pattern):
    return lambda text: pattern.search(text) is not None


# ---------------------------------------------------------------------------
# D1 - communication style
# ---------------------------------------------------------------------------


def words_per_turn(convo: dict) -> float:
    """Wds/trn: mean words per customer turn."""
    texts = customer_texts(convo)
    return float(np.mean([n_words(t) for t in texts])) if texts else float("nan")


def short_turn_rate(convo: dict) -> float:
    """Short%: share of customer turns with <= 3 words."""
    return _rate(customer_texts(convo), lambda t: n_words(t) <= 3)


def polite_rate(convo: dict) -> float:
    """Polite%: share of turns with please / thanks / sorry."""
    return _rate(customer_texts(convo), _matches(POLITE))


def dash_rate(convo: dict) -> float:
    """Dash%: share of turns with an em or en dash (an LLM tell)."""
    return _rate(customer_texts(convo), _matches(DASH))


def ack_rate(convo: dict) -> float:
    """Ack%: share of turns that are ONLY an acknowledgement ("ok", "sure", "got it")."""
    return _rate(customer_texts(convo), lambda t: ACK_ONLY.match(t) is not None)


def verbosity_cv(convo: dict) -> float:
    """VerbCV: std / mean of customer turn lengths within the conversation."""
    lengths = [n_words(t) for t in customer_texts(convo)]
    if len(lengths) < 2 or np.mean(lengths) == 0:
        return float("nan")
    return float(np.std(lengths) / np.mean(lengths))


def repeat_flag(convo: dict, min_count: int = 6) -> float:
    """Repeat%: 1 if some word 3-gram occurs > 5 times in the customer turns."""
    counts: Counter = Counter()
    for text in customer_texts(convo):
        w = text.lower().split()
        counts.update(zip(w, w[1:], w[2:]))
    return float(any(c >= min_count for c in counts.values()))


def identity_confusion(convo: dict) -> float:
    """IDConf%: 1 if the customer uses agent-like service language (role confusion)."""
    return float(any(AGENT_LANGUAGE.search(t) for t in customer_texts(convo)))


# ---------------------------------------------------------------------------
# D2 - information pattern
# ---------------------------------------------------------------------------


def front_loading(convo: dict) -> float:
    """FrntId%: share of all customer words that fall in the first two customer turns."""
    lengths = [n_words(t) for t in customer_texts(convo)]
    total = sum(lengths)
    return sum(lengths[:2]) / total if total else float("nan")


def ids_per_turn(convo: dict) -> float:
    """IDs/trn: mean number of identifier-like tokens per customer turn."""
    texts = customer_texts(convo)
    return float(np.mean([len(ID_TOKEN.findall(t)) for t in texts])) if texts else float("nan")


def opening_words(convo: dict) -> float:
    """Open.Wds: word count of the first customer turn."""
    texts = customer_texts(convo)
    return float(n_words(texts[0])) if texts else float("nan")


# ---------------------------------------------------------------------------
# D3 - clarification (priority: pushback > clarification > info question)
# ---------------------------------------------------------------------------


def classify_question(text: str) -> str | None:
    if PUSHBACK.search(text):
        return "pushback"
    if CLARIFY.search(text):
        return "clarify"
    if "?" in text or INFO_Q_START.match(text):
        return "info"
    return None


def uncertainty_rate(convo: dict) -> float:
    """Uncert%: share of turns with hedging ("maybe", "not sure", "I think")."""
    return _rate(customer_texts(convo), _matches(UNCERTAIN))


def certainty_rate(convo: dict) -> float:
    """Certn%: share of turns with certainty markers ("definitely", "for sure")."""
    return _rate(customer_texts(convo), _matches(CERTAIN))


def pushback_rate(convo: dict) -> float:
    """Pushbk%: share of turns that push back ("are you sure?", "you already asked")."""
    return _rate(customer_texts(convo), lambda t: classify_question(t) == "pushback")


def clarify_q_rate(convo: dict) -> float:
    """ClarfyQ%: share of clarification turns (excluding pushback)."""
    return _rate(customer_texts(convo), lambda t: classify_question(t) == "clarify")


def info_q_rate(convo: dict) -> float:
    """InfoQ%: share of information-seeking turns (excluding the two above)."""
    return _rate(customer_texts(convo), lambda t: classify_question(t) == "info")


# ---------------------------------------------------------------------------
# D4 - error reaction (over all customer turns, like the paper)
# ---------------------------------------------------------------------------


def emotion_rate(convo: dict) -> float:
    """Emot%: share of turns with frustration markers ("frustrated", "ugh")."""
    return _rate(customer_texts(convo), _matches(EMOTION))


def accuse_rate(convo: dict) -> float:
    """Accuse%: share of turns with accusatory language ("useless", "scam")."""
    return _rate(customer_texts(convo), _matches(ACCUSE))


def pivot_rate(convo: dict) -> float:
    """Pivot%: share of turns signalling a change of strategy ("instead", "never mind")."""
    return _rate(customer_texts(convo), _matches(PIVOT))


# ---------------------------------------------------------------------------
# Extras: outcome and goal metrics (ours, not in the paper)
# ---------------------------------------------------------------------------


def resolved_closure(convo: dict, last_k: int = 2) -> float:
    """1 if the customer closes positively ("thanks, that solved it") without anger."""
    tail = customer_texts(convo)[-last_k:]
    if not tail:
        return float("nan")
    positive = any(RESOLVED_CLOSURE.search(t) for t in tail)
    negative = any(ACCUSE.search(t) or EMOTION.search(t) for t in tail)
    return float(positive and not negative)


def abandoned(convo: dict) -> float:
    """1 if the customer ended the chat without a resolved closure.

    Simulated runs carry ``end_reason``; a run cut by ``max_turns`` is not an
    abandonment. Human logs have no end_reason, so we assume the customer ended it.
    """
    if convo.get("end_reason") not in (None, "customer_end"):
        return 0.0
    closure = resolved_closure(convo)
    return float("nan") if np.isnan(closure) else 1.0 - closure


def _norm(value) -> str:
    return str(value).lower().replace("_", " ").strip()


def scenario_slot_values(scenario: dict) -> list[str]:
    """Values from an ABCD scenario a goal-keeping customer is expected to mention."""
    personal = scenario.get("personal", {})
    order = scenario.get("order", {})
    values = [personal.get("customer_name"), personal.get("username"),
              personal.get("email"), order.get("order_id")]
    for name in scenario.get("product", {}).get("names", []):
        values.append(name)  # e.g. "michael_kors jeans" -> "michael kors jeans"
    return [_norm(v) for v in values if v]


def goal_adherence(convo: dict) -> float:
    """Share of scenario slot values that appear in at least one customer turn.

    Not every slot is needed in every flow, so the absolute level is only
    meaningful relative to the human reference on the same scenarios (H5 / R1).
    """
    scenario = convo.get("scenario")
    if not scenario:
        return float("nan")
    values = scenario_slot_values(scenario)
    if not values:
        return float("nan")
    text = _norm(" ".join(customer_texts(convo)))
    hits = []
    for v in values:
        # products: the product type or the brand alone counts as a mention
        parts = [v] if " " not in v or "@" in v else [v] + v.split()
        hits.append(any(p in text for p in parts if len(p) > 2))
    return float(np.mean(hits))


def offscript_id_rate(convo: dict) -> float:
    """Share of customer turns containing an ID-like token NOT in the scenario (hallucinated)."""
    scenario = convo.get("scenario")
    texts = customer_texts(convo)
    if not scenario or not texts:
        return float("nan")
    known = " ".join(scenario_slot_values(scenario) + [
        _norm(v) for part in ("personal", "order") for v in scenario.get(part, {}).values()
    ])
    return _rate(texts, lambda t: any(tok.lower() not in known for tok in ID_TOKEN.findall(t)))


# ---------------------------------------------------------------------------
# Registry: name -> (dimension, function). "X" = extras, not in the composite.
# ---------------------------------------------------------------------------

METRICS = {
    "wds_per_turn": ("D1", words_per_turn),
    "short_pct": ("D1", short_turn_rate),
    "polite_pct": ("D1", polite_rate),
    "dash_pct": ("D1", dash_rate),
    "ack_pct": ("D1", ack_rate),
    "verb_cv": ("D1", verbosity_cv),
    "repeat_pct": ("D1", repeat_flag),
    "idconf_pct": ("D1", identity_confusion),
    "frontid_pct": ("D2", front_loading),
    "ids_per_turn": ("D2", ids_per_turn),
    "open_wds": ("D2", opening_words),
    "uncert_pct": ("D3", uncertainty_rate),
    "certn_pct": ("D3", certainty_rate),
    "pushbk_pct": ("D3", pushback_rate),
    "clarfyq_pct": ("D3", clarify_q_rate),
    "infoq_pct": ("D3", info_q_rate),
    "emot_pct": ("D4", emotion_rate),
    "accuse_pct": ("D4", accuse_rate),
    "pivot_pct": ("D4", pivot_rate),
    "abandon_pct": ("X", abandoned),
    "resolved_pct": ("X", resolved_closure),
    "goal_adherence": ("X", goal_adherence),
    "offscript_pct": ("X", offscript_id_rate),
}
COMPOSITE_DIMS = ("D1", "D2", "D3", "D4")


def compute_metrics(convos: list[dict]) -> dict[str, np.ndarray]:
    """Per-conversation values for every metric: {name: array of len(convos)} (may hold NaN)."""
    return {name: np.array([fn(c) for c in convos], dtype=float) for name, (_, fn) in METRICS.items()}


# ---------------------------------------------------------------------------
# Distribution similarity
# ---------------------------------------------------------------------------


def dice_similarity(sim: float, human: float) -> float:
    """Sorensen-Dice on aggregate values, 0-100 (paper): 2*min / (sum) * 100."""
    if np.isnan(sim) or np.isnan(human):
        return float("nan")
    if sim == 0 and human == 0:
        return 100.0
    return float(200.0 * min(sim, human) / (sim + human))


def wasserstein_1d(u, v) -> float:
    """Earth mover's distance between two 1-D samples (same algorithm as scipy)."""
    u, v = np.sort(np.asarray(u, float)), np.sort(np.asarray(v, float))
    if len(u) == 0 or len(v) == 0:
        return float("nan")
    grid = np.sort(np.concatenate([u, v]))
    cdf_u = np.searchsorted(u, grid[:-1], side="right") / len(u)
    cdf_v = np.searchsorted(v, grid[:-1], side="right") / len(v)
    return float(np.sum(np.abs(cdf_u - cdf_v) * np.diff(grid)))


def _clean(x: np.ndarray) -> np.ndarray:
    return x[~np.isnan(x)]


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def compare_values(sim_vals: dict, hum_vals: dict) -> dict:
    """Compare per-conversation metric values of a simulator corpus and a human corpus.

    Returns per-metric aggregates, Dice, Wasserstein (raw and divided by the
    human std so metrics share one scale), per-dimension Dice means, and a
    0-100 composite = mean of D1..D4 (the paper's USI without the
    evaluation-alignment terms, which need survey data we do not have).
    """
    rows = {}
    for name, (dim, _) in METRICS.items():
        s, h = _clean(sim_vals[name]), _clean(hum_vals[name])
        w1 = wasserstein_1d(s, h)
        std = float(np.std(h)) if len(h) else float("nan")
        rows[name] = {
            "dim": dim, "sim": _mean(s), "human": _mean(h),
            "dice": dice_similarity(_mean(s), _mean(h)),
            "w1": w1, "w1_norm": w1 / std if std > 0 else float("nan"),
        }
    dims = {}
    for d in COMPOSITE_DIMS:
        scores = [r["dice"] for r in rows.values() if r["dim"] == d and not np.isnan(r["dice"])]
        dims[d] = float(np.mean(scores)) if scores else float("nan")
    return {"metrics": rows, "dimensions": dims, "composite": float(np.nanmean(list(dims.values())))}


def compare(sim_convos: list[dict], human_convos: list[dict]) -> dict:
    return compare_values(compute_metrics(sim_convos), compute_metrics(human_convos))


def composite_score(sim_convos: list[dict], human_convos: list[dict]) -> float:
    return compare(sim_convos, human_convos)["composite"]


def human_ceiling(human_convos: list[dict], n_splits: int = 10, seed: int = 0) -> dict:
    """Human-vs-human ceiling: split human data in random halves and compare them.

    A simulator cannot be expected to beat this score; report it next to every
    simulator score (R7).
    """
    rng = np.random.default_rng(seed)
    values = compute_metrics(human_convos)
    scores = []
    for _ in range(n_splits):
        idx = rng.permutation(len(human_convos))
        a, b = idx[: len(idx) // 2], idx[len(idx) // 2 :]
        half_a = {k: v[a] for k, v in values.items()}
        half_b = {k: v[b] for k, v in values.items()}
        scores.append(compare_values(half_a, half_b)["composite"])
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "scores": scores}
