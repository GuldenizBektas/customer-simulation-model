# usersim: can a small fine-tuned model act like a real customer?

LLM-based "user simulators" are used to test support agents, but they tend to be
too cooperative: they dump every detail in the first message, never get annoyed
and politely pivot when the agent is wrong. The result is an optimistic
picture of agent quality ("easy mode"; see Zhou et al., *Mind the Sim2Real Gap
in User Simulation for Agentic Tasks*, arXiv:2603.11245).

This repo tests a simple idea: fine-tune a small model (Qwen3-4B-Instruct-2507,
QLoRA with Unsloth) on the **customer side** of human support chats (ABCD), and
check whether it

1. behaves more like human customers than the same model with a prompt (H1, H2),
2. gives agent success rates closer to those measured with humans (H3),
3. separates a good support agent from broken ones more clearly (H4),
4. without getting its "realism" from simply forgetting its goal (H5).

Negative results will be published too.

## How it works

```
ABCD chats ──prepare_data──> SFT JSONL (system = scenario, user = agent, assistant = customer)
                                 │
                               train (QLoRA, loss on customer turns only)
                                 │
simulators (ft, prompted_basic, prompted_realistic, ...) × agents (good, broken_*) × scenarios × seeds
                                 │ simulate
                           transcripts.jsonl
                                 │ evaluate
          behavior metrics (D1-D4) vs humans · success rate · good-vs-broken effect size · H5 · leakage
```

Metrics are regex/count based (no LLM judge): words per turn, short turns,
politeness, dashes, pure acknowledgements, verbosity variation, repetition,
role confusion, front-loading of information, IDs per turn, opening length,
hedging/certainty, pushback/clarification/info questions, emotion, accusation,
strategy pivots, plus abandonment, resolved closure, goal adherence and
off-script (hallucinated) IDs. Similarity to humans uses Sørensen–Dice on
aggregates (0-100, as in the paper) and Wasserstein distance on
per-conversation distributions, next to a human-vs-human ceiling.

## Setup

Laptop (data prep, metrics, local simulation with Ollama, evaluation):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # numpy, scipy, pandas, pyyaml, ollama, pytest
pytest
```

GPU (Kaggle T4) for training: `pip install -e ".[train]"` (unsloth, trl,
transformers, datasets, peft). Do not install these on a Mac; use
`notebooks/kaggle_train.ipynb`.

Local models: install [Ollama](https://ollama.com) and
`ollama pull qwen3:4b-instruct-2507-q4_K_M`.

All commands run from the repo root. `data/` and `outputs/` are git-ignored.

## Pipeline (in order)

| Step | Command | Where |
|---|---|---|
| 1. Get data | download `abcd_v1.1.json(.gz)` and `kb.json` from [asappresearch/abcd](https://github.com/asappresearch/abcd) into `data/raw/` | laptop |
| 2. Prepare SFT data + grouped split | `python -m usersim.prepare_data --config configs/data.yaml` | laptop |
| 3. Token lengths, set `max_seq_len` | cell 4 of `notebooks/kaggle_train.ipynb` | Kaggle |
| 4. Human reference values + ceiling | `python -c "from usersim import metrics as M, evaluate as E; h=E.load_jsonl('data/processed/test.jsonl'); print(M.human_ceiling(h))"` | laptop |
| 5. Baseline B1 smoke run (20 chats) | `python -m usersim.simulate --config configs/simulate.yaml --run-id smoke --n-scenarios 20` | laptop (Ollama) |
| 6. Training smoke test (500 ex, 1 epoch) | `python -m usersim.train --config configs/train.yaml --max-examples 500 --epochs 1` | Kaggle |
| 7. Full training + GGUF | `python -m usersim.train --config configs/train.yaml --export-gguf` | Kaggle |
| 8. Run the fine-tuned model locally | `ollama create usersim-qwen3-4b -f Modelfile` (see below), set `simulators.ft.enabled: true` | laptop |
| 9. Small simulation: ft / B1 / B2 × good / broken | `python -m usersim.simulate --config configs/simulate.yaml --run-id small` | laptop |
| 10. Large run (≥150 scenarios × 3 seeds) | `notebooks/kaggle_simulate.ipynb` | Kaggle |
| 11. Evaluate (+ tau-usi, leakage) | `python -m usersim.evaluate --config configs/eval.yaml` → `outputs/eval/results.md` | laptop |

Modelfile for step 8 (GGUF from step 7):

```
FROM ./outputs/train/qwen3-4b-usersim/gguf/<file>.gguf   # TODO: exact file name
TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
```

## Repository layout

```
configs/      data.yaml, train.yaml, simulate.yaml, eval.yaml
src/usersim/
  prepare_data.py   ABCD -> customer-side SFT JSONL, PII masking, grouped split
  train.py          Unsloth + TRL SFT, loss on customer turns only (GPU)
  agents.py         good / broken support agents, Ollama backend, rule-based agent
  simulators.py     prompted (basic, realistic), fine-tuned, UserLM (TODO)
  simulate.py       simulator × agent × scenario × seed loop, resumable
  metrics.py        D1-D4 metrics, extras, Dice / Wasserstein, human ceiling
  evaluate.py       H1-H5 tables, bootstrap CIs, leakage check
  tau_usi.py        loader for cmu-lti/tau-usi (EVALUATION ONLY)
notebooks/    kaggle_train.ipynb, kaggle_simulate.ipynb
tests/        pytest with fake ABCD-shaped data
```

## Design choices worth knowing

- **Roles are flipped.** The model is trained as the *assistant* that plays the
  customer; agent turns are the *user*. Loss is only on customer tokens.
- **Split by intent (subflow), not by conversation.** Test scenarios come from
  intents never seen in training, so a good score cannot come from memorising
  scenario templates. `group_by` in `configs/data.yaml` can relax this.
- **The agent never sees the customer's goal.** It marks a finished task with
  `[RESOLVED: <subflow>]`; the tag is hidden from the customer and used for a
  rule-based success rate.
- **Broken agents** are prompt variants (wrong policy, over-asking, polite but
  useless) plus two deterministic ones (looping, premature "resolved"), so at
  least some faults happen for sure. `RuleBasedAgent` needs no LLM at all.
- **Composite score** = mean of D1-D4 Dice scores. The paper's USI also
  includes evaluation-alignment terms that need survey data; we leave them out.
- **Lexicons are short and explicit** (top of `metrics.py`). They are easy to
  game (R7), which is why we also report full distributions (Wasserstein) and
  the human-vs-human ceiling.

## Data and licenses

- ABCD (ASAPP, NAACL 2021): customers are crowdworkers role-playing a given
  scenario, not real customers. Repo license MIT; TODO confirm data terms.
- `cmu-lti/tau-usi`: CC BY-NC 4.0 with an explicit ban on using it to train or
  improve any LLM. Used here **only** as out-of-domain evaluation data.
- Raw data is never committed or redistributed.

## Reproduce

- [ ] TODO: pin package versions after the first Kaggle run (`pip freeze`).
- [ ] TODO: confirm the ABCD download paths and `kb.json` format.
- [ ] TODO: fill `max_seq_len` from token-length stats.
- [ ] TODO: link the Hugging Face adapter and GGUF.
- [ ] TODO: set `human_reference_success` in `configs/eval.yaml` and document its source.
- [ ] TODO: one command (or notebook) that runs steps 2-11 end to end.
- [ ] TODO: add the final results table.

## Citation

If you use the metrics, cite Zhou et al., *Mind the Sim2Real Gap in User
Simulation for Agentic Tasks*, COLM 2026 (arXiv:2603.11245), and ABCD
(Chen et al., NAACL 2021, arXiv:2104.00783).

## License

MIT (code). Data and models keep their own licenses.
# customer-simulation-model
# customer-simulation-model
