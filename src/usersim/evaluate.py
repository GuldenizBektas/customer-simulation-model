"""Evaluate simulator runs against human conversations.

(a) Behavioral distance: simulator vs held-out human customers (H1), with the
    human-vs-human ceiling; optionally vs tau-usi humans (H2).
(b) Agent success rate and optimism gap (H3).
(c) Discrimination: good vs broken agent, Cohen's d + bootstrap 95% CI (H4).
(d) H5 checks: goal adherence, role confusion, off-script IDs vs the baseline.
(e) Leakage: n-gram overlap between generated customer turns and the training set.

Writes outputs/eval/results.json and results.md.

Usage:
    python -m usersim.evaluate --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import metrics as M


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def cohens_d(a, b) -> float:
    """Effect size of mean(b) - mean(a) using the pooled standard deviation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    diff = b.mean() - a.mean()
    if pooled == 0:  # no variance at all: perfectly separated or identical
        return 0.0 if diff == 0 else float(np.sign(diff) * np.inf)
    return float(diff / pooled)


def bootstrap_ci(stat, *samples, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> list[float]:
    """Percentile CI of ``stat(*resampled)``; each sample is resampled independently."""
    rng = np.random.default_rng(seed)
    samples = [np.asarray(s) for s in samples]
    values = [stat(*[s[rng.integers(0, len(s), len(s))] for s in samples]) for _ in range(n)]
    return [float(np.nanpercentile(values, 100 * alpha / 2)), float(np.nanpercentile(values, 100 * (1 - alpha / 2)))]


# ---------------------------------------------------------------------------
# Per-conversation outcomes
# ---------------------------------------------------------------------------


def agent_success(convo: dict) -> float:
    """1 if the agent resolved the RIGHT subflow and the customer did not abandon."""
    target = convo["scenario"]["subflow"]
    resolved = any(t.get("resolved") == target for t in convo["turns"] if t["role"] == "agent")
    return float(resolved and M.abandoned(convo) != 1.0)


def frustration(convo: dict) -> float:
    return np.nanmean([M.emotion_rate(convo), M.accuse_rate(convo)])


OUTCOMES = {"success": agent_success, "abandon": M.abandoned, "frustration": frustration}


# ---------------------------------------------------------------------------
# (a) behavioral distance
# ---------------------------------------------------------------------------


def behavioral(sim_convos, human_convos, n_boot, seed) -> dict:
    sim_vals, hum_vals = M.compute_metrics(sim_convos), M.compute_metrics(human_convos)
    result = M.compare_values(sim_vals, hum_vals)

    def composite(si, hi):
        return M.compare_values({k: v[si] for k, v in sim_vals.items()},
                                {k: v[hi] for k, v in hum_vals.items()})["composite"]

    result["composite_ci"] = bootstrap_ci(composite, np.arange(len(sim_convos)),
                                          np.arange(len(human_convos)), n=n_boot, seed=seed)
    return result


# ---------------------------------------------------------------------------
# (e) leakage
# ---------------------------------------------------------------------------


def ngrams(text: str, n: int) -> set:
    w = text.lower().split()
    return {tuple(w[i: i + n]) for i in range(len(w) - n + 1)}


def ngram_overlap(convos, train_ngrams: set, n: int) -> float:
    """Share of customer n-grams that also occur in the training customer turns."""
    grams = [g for c in convos for t in M.customer_texts(c) for g in ngrams(t, n)]
    return float(np.mean([g in train_ngrams for g in grams])) if grams else float("nan")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(cfg: dict) -> dict:
    humans = load_jsonl(cfg["human_file"])
    runs = [r for d in cfg["run_dirs"] for r in load_jsonl(Path(d) / "transcripts.jsonl")]
    sims = sorted({r["simulator"] for r in runs})
    agents = sorted({r["agent"] for r in runs})
    good, n_boot, seed = cfg["good_agent"], cfg["bootstrap"], cfg["seed"]
    by = lambda s, a: [r for r in runs if r["simulator"] == s and r["agent"] == a]

    res = {"n_human": len(humans), "simulators": sims, "agents": agents}

    # (a) H1: behavior with the good agent vs held-out humans, plus ceiling
    res["ceiling"] = M.human_ceiling(humans, cfg["ceiling_splits"], seed)
    res["behavior"] = {s: behavioral(by(s, good), humans, n_boot, seed) for s in sims}
    if cfg.get("tau_usi_file"):  # H2, out-of-domain humans (evaluation only)
        from .tau_usi import load_tau_usi

        tau = load_tau_usi(cfg["tau_usi_file"])
        res["behavior_tau_usi"] = {s: M.compare(by(s, good), tau)["composite"] for s in sims}

    # (b) H3: success rate and optimism gap
    ref = cfg.get("human_reference_success")
    res["success"] = {}
    for s in sims:
        for a in agents:
            vals = [agent_success(c) for c in by(s, a)]
            keeps_goal = [agent_success(c) for c in by(s, a)
                          if M.goal_adherence(c) >= cfg["goal_keeping_min"] and M.offscript_id_rate(c) == 0]
            res["success"][f"{s}|{a}"] = {
                "rate": float(np.mean(vals)) if vals else float("nan"),
                "ci": bootstrap_ci(np.mean, vals, n=n_boot, seed=seed) if vals else None,
                "rate_goal_keeping": float(np.mean(keeps_goal)) if keeps_goal else float("nan"),
                "optimism_gap": (float(np.mean(vals)) - ref) if (vals and ref is not None) else None,
            }

    # (c) H4: discrimination good vs each broken agent
    res["discrimination"] = {}
    for s in sims:
        for a in agents:
            if a == good:
                continue
            for name, fn in OUTCOMES.items():
                x = [v for v in (fn(c) for c in by(s, good)) if not np.isnan(v)]
                y = [v for v in (fn(c) for c in by(s, a)) if not np.isnan(v)]
                res["discrimination"][f"{s}|{a}|{name}"] = {
                    "good": float(np.mean(x)) if x else float("nan"),
                    "broken": float(np.mean(y)) if y else float("nan"),
                    "cohens_d": cohens_d(x, y),
                    "ci": bootstrap_ci(cohens_d, x, y, n=n_boot, seed=seed) if len(x) > 1 and len(y) > 1 else None,
                }

    # (d) H5: does "realism" come from forgetting the goal / role confusion?
    h5 = {}
    for s in sims:
        convos = by(s, good)
        h5[s] = {m: float(np.nanmean([fn(c) for c in convos])) if convos else float("nan")
                 for m, fn in (("goal_adherence", M.goal_adherence), ("idconf", M.identity_confusion),
                               ("offscript", M.offscript_id_rate))}
    base = h5.get(cfg["baseline_simulator"], {}).get("goal_adherence", float("nan"))
    for s in h5:
        h5[s]["h5_violation"] = bool(base - h5[s]["goal_adherence"] > cfg["h5_max_adherence_drop"])
    h5["human"] = {"goal_adherence": float(np.nanmean([M.goal_adherence(c) for c in humans]))}
    res["h5"] = h5

    # (e) leakage
    n = cfg["leakage_ngram"]
    train_grams = {g for c in load_jsonl(cfg["train_file"]) for t in M.customer_texts(c) for g in ngrams(t, n)}
    res["leakage"] = {s: ngram_overlap([r for r in runs if r["simulator"] == s], train_grams, n) for s in sims}
    res["leakage"]["human_test (reference)"] = ngram_overlap(humans, train_grams, n)
    return res


def to_markdown(res: dict) -> str:
    f = lambda x: "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
    lines = ["# Results", "", f"Human conversations: {res['n_human']}  ",
             f"Human-human ceiling (composite): {f(res['ceiling']['mean'])} +/- {f(res['ceiling']['std'])}", "",
             "## (a) Behavioral similarity to humans (good agent, 0-100, higher = closer)", "",
             "| simulator | D1 | D2 | D3 | D4 | composite | 95% CI |", "|---|---|---|---|---|---|---|"]
    for s, b in res["behavior"].items():
        d = b["dimensions"]
        lines.append(f"| {s} | {f(d['D1'])} | {f(d['D2'])} | {f(d['D3'])} | {f(d['D4'])} | "
                     f"{f(b['composite'])} | [{f(b['composite_ci'][0])}, {f(b['composite_ci'][1])}] |")
    lines += ["", "## (b) Agent success rate", "", "| simulator / agent | rate | 95% CI | goal-keeping only | optimism gap |",
              "|---|---|---|---|---|"]
    for k, v in res["success"].items():
        ci = v["ci"] or [None, None]
        lines.append(f"| {k} | {f(v['rate'])} | [{f(ci[0])}, {f(ci[1])}] | {f(v['rate_goal_keeping'])} | {f(v['optimism_gap'])} |")
    lines += ["", "## (c) Discrimination: good vs broken agent (Cohen's d)", "",
              "| simulator / agent / outcome | good | broken | d | 95% CI |", "|---|---|---|---|---|"]
    for k, v in res["discrimination"].items():
        ci = v["ci"] or [None, None]
        lines.append(f"| {k} | {f(v['good'])} | {f(v['broken'])} | {f(v['cohens_d'])} | [{f(ci[0])}, {f(ci[1])}] |")
    lines += ["", "## (d) H5 checks", "", "| simulator | goal adherence | IDConf | off-script IDs | H5 violation |",
              "|---|---|---|---|---|"]
    for s, v in res["h5"].items():
        lines.append(f"| {s} | {f(v['goal_adherence'])} | {f(v.get('idconf'))} | {f(v.get('offscript'))} | {v.get('h5_violation', '-')} |")
    lines += ["", "## (e) Leakage: share of customer n-grams found in training data", "",
              "| source | overlap |", "|---|---|"]
    lines += [f"| {k} | {f(v)} |" for k, v in res["leakage"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate simulator runs against human conversations.")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args(argv)
    import yaml

    cfg = yaml.safe_load(open(args.config))
    res = evaluate(cfg)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    (out / "results.md").write_text(to_markdown(res))
    print(to_markdown(res))


if __name__ == "__main__":
    main()
