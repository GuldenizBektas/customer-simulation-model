"""Run simulator x agent x scenario x seed conversations and save transcripts.

Output: outputs/runs/<run_id>/transcripts.jsonl, one conversation per line:
    {"simulator", "agent", "scenario_id", "seed", "end_reason", "turns", "scenario", ...}

Re-running the same command resumes: finished (simulator, agent, scenario, seed)
combinations are skipped.

Usage:
    python -m usersim.simulate --config configs/simulate.yaml
    python -m usersim.simulate --config configs/simulate.yaml --n-scenarios 20 --run-id smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .agents import (AGENT_CLASSES, GREETING, HFBackend, OllamaBackend, RuleBasedAgent,
                     VLLMBackend, load_policy, split_resolution)
from .simulators import FineTunedSimulator, PromptedSimulator, UserLMSimulator


def run_conversation(simulator, agent, scenario: dict, max_turns: int, seed: int | None = None):
    """One chat. The agent greets, then customer and agent alternate.

    Returns (turns, end_reason) with end_reason in {"customer_end", "max_turns"}.
    """
    history = [{"role": "agent", "text": GREETING, "resolved": None}]
    for _ in range(max_turns):
        text, ended = simulator.respond(scenario, history, seed=seed)
        if text:
            history.append({"role": "customer", "text": text})
        if ended:
            return history, "customer_end"
        visible, resolved = split_resolution(agent.respond(history, scenario, seed=seed))
        history.append({"role": "agent", "text": visible, "resolved": resolved})
    return history, "max_turns"


# ---------------------------------------------------------------------------
# Building simulators / agents from the config
# ---------------------------------------------------------------------------


def make_backend(spec: dict):
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "ollama":
        return OllamaBackend(spec.pop("model"), spec.pop("host", "http://localhost:11434"), **spec)
    if kind == "hf":
        return HFBackend(**spec)
    if kind == "vllm":
        return VLLMBackend(**spec)
    raise ValueError(f"unknown backend type: {kind}")


def make_simulator(spec: dict, backends: dict):
    if spec["class"] == "userlm":
        return UserLMSimulator(backends[spec["backend"]])
    backend = backends[spec["backend"]]
    if spec["class"] == "ft":
        return FineTunedSimulator(backend)
    return PromptedSimulator(backend, style=spec.get("style", "basic"))


def make_agent(name: str, spec: dict, backends: dict, policy: str):
    if spec["class"] == "rule":
        return RuleBasedAgent(fault=spec.get("fault"))
    agent = AGENT_CLASSES[spec["class"]](backends[spec["backend"]], policy)
    agent.name = name
    return agent


def load_scenarios(path: str, n: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in open(path)]
    random.Random(seed).shuffle(rows)
    return [{"id": r["convo_id"], "scenario": r["scenario"]} for r in rows[:n]]


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    for line in open(path):
        r = json.loads(line)
        keys.add((r["simulator"], r["agent"], r["scenario_id"], r["seed"]))
    return keys


def run(cfg: dict) -> Path:
    out_dir = Path(cfg["out_root"]) / cfg["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    out_file = out_dir / "transcripts.jsonl"
    done = done_keys(out_file)

    needed = {s["backend"] for s in cfg["simulators"].values() if s.get("enabled", True) and "backend" in s}
    needed |= {a["backend"] for a in cfg["agents"].values() if a.get("enabled", True) and "backend" in a}
    backends = {name: make_backend(cfg["backends"][name]) for name in needed}
    policy = load_policy(cfg.get("kb_path"))
    simulators = {n: make_simulator(s, backends) for n, s in cfg["simulators"].items() if s.get("enabled", True)}
    agents = {n: make_agent(n, a, backends, policy) for n, a in cfg["agents"].items() if a.get("enabled", True)}
    scenarios = load_scenarios(cfg["scenarios_file"], cfg["n_scenarios"], cfg["seed"])

    with open(out_file, "a") as f:
        for sim_name, sim in simulators.items():
            for agent_name, agent in agents.items():
                for sc in scenarios:
                    for seed in cfg["seeds"]:
                        key = (sim_name, agent_name, sc["id"], seed)
                        if key in done:
                            continue
                        start = time.time()
                        try:
                            turns, end_reason = run_conversation(sim, agent, sc["scenario"], cfg["max_turns"], seed)
                        except Exception as err:  # keep going; the combination is retried on resume
                            print(f"[error] {key}: {err}")
                            continue
                        record = {
                            "run_id": cfg["run_id"], "simulator": sim_name, "agent": agent_name,
                            "scenario_id": sc["id"], "seed": seed, "end_reason": end_reason,
                            "n_turns": len(turns), "seconds": round(time.time() - start, 1),
                            "turns": turns, "scenario": sc["scenario"],
                        }
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                        print(f"[done] {key} {end_reason} {len(turns)} turns")
    return out_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run simulated customer-agent conversations.")
    parser.add_argument("--config", default="configs/simulate.yaml")
    parser.add_argument("--run-id", help="override run_id (output folder name)")
    parser.add_argument("--n-scenarios", type=int, help="override number of scenarios")
    args = parser.parse_args(argv)
    import yaml

    cfg = yaml.safe_load(open(args.config))
    if args.run_id:
        cfg["run_id"] = args.run_id
    if args.n_scenarios:
        cfg["n_scenarios"] = args.n_scenarios
    print("transcripts:", run(cfg))


if __name__ == "__main__":
    main()
