"""Support agents the simulated customers talk to: one good agent, several broken ones.

The agent under test is FIXED across all simulators, so differences in the
results come from the simulator, not from the agent.

An agent signals that it finished the customer's request by writing
``[RESOLVED: <subflow>]``. The tag is stripped before the customer sees the
message and is used for the rule-based agent success rate.

Backends are behind a tiny interface (``Backend.chat``) so the same agent can
run on local Ollama or on Kaggle (HF / vLLM). No paid APIs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

GREETING = "Hi! How can I help you today?"
RESOLVED_TAG = re.compile(r"\[RESOLVED:\s*([\w-]+)\]")


def split_resolution(text: str) -> tuple[str, str | None]:
    """Return (visible text, resolved subflow or None)."""
    match = RESOLVED_TAG.search(text)
    return RESOLVED_TAG.sub("", text).strip(), (match.group(1) if match else None)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class Backend:
    """Anything that maps chat messages to a reply string."""

    def chat(self, messages: list[dict], seed: int | None = None) -> str:
        raise NotImplementedError


class OllamaBackend(Backend):
    """Local model served by Ollama, e.g. ``qwen3:4b-instruct-2507-q4_K_M``."""

    def __init__(self, model: str, host: str = "http://localhost:11434", **options):
        import ollama  # light client; imported lazily so tests do not need it

        self.client = ollama.Client(host=host)
        self.model = model
        self.options = options  # temperature, top_p, num_predict, ...

    def chat(self, messages, seed=None):
        options = dict(self.options, **({"seed": seed} if seed is not None else {}))
        response = self.client.chat(model=self.model, messages=messages, options=options)
        return response["message"]["content"]


class HFBackend(Backend):
    """transformers (+ optional PEFT adapter) on a Kaggle GPU."""

    def __init__(self, model: str, adapter: str | None = None, **generation_kwargs):
        # TODO(kaggle): load with unsloth.FastLanguageModel.from_pretrained(model, load_in_4bit=True),
        #   then PeftModel.from_pretrained(model, adapter) if adapter; FastLanguageModel.for_inference(model).
        raise NotImplementedError("HFBackend runs on Kaggle; see notebooks/kaggle_simulate.ipynb")

    def chat(self, messages, seed=None):
        # TODO(kaggle): tokenizer.apply_chat_template(..., add_generation_prompt=True) -> generate -> decode
        raise NotImplementedError


class VLLMBackend(Backend):
    """vLLM offline engine for large batched runs on Kaggle."""

    def __init__(self, model: str, adapter: str | None = None, **sampling_kwargs):
        # TODO(kaggle): vllm.LLM(model, enable_lora=bool(adapter)); SamplingParams(**sampling_kwargs)
        raise NotImplementedError("VLLMBackend runs on Kaggle; see notebooks/kaggle_simulate.ipynb")

    def chat(self, messages, seed=None):
        # TODO(kaggle): llm.chat(messages, sampling_params, lora_request=...)
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Prompted agents
# ---------------------------------------------------------------------------

AGENT_PROMPT = """You are a customer support agent for an online clothing store.
Follow the company policy below. Verify the customer's identity before account changes,
ask only for the information you need, and be concise.
When you have fully handled the request, include the tag [RESOLVED: <subflow>] in your
message, where <subflow> is one of the policy keys below.

Policy (subflow: required steps):
{policy}"""

DEFAULT_POLICY = "(policy not loaded; TODO: point configs/simulate.yaml at ABCD kb.json)"


def load_policy(kb_path: str | None) -> str:
    """Render ABCD ``kb.json`` (assumed: {subflow: [action, ...]}) as text.

    TODO(verify): check the real kb.json / guidelines.json format after download.
    """
    if not kb_path or not Path(kb_path).exists():
        return DEFAULT_POLICY
    kb = json.loads(Path(kb_path).read_text())
    return "\n".join(f"- {subflow}: {' -> '.join(steps)}" for subflow, steps in kb.items())


class Agent:
    """Base agent. ``respond`` sees the history so far and returns the next agent message."""

    name = "agent"

    def respond(self, history: list[dict], scenario: dict, seed: int | None = None) -> str:
        raise NotImplementedError


class GoodAgent(Agent):
    """Follows the ABCD policy. It never sees the customer's goal, only the chat."""

    name = "good"
    fault = ""  # broken variants override this

    def __init__(self, backend: Backend, policy: str = DEFAULT_POLICY):
        self.backend = backend
        self.system = AGENT_PROMPT.format(policy=policy)
        if self.fault:
            self.system += f"\n\nIMPORTANT behaviour for this role: {self.fault}"

    def respond(self, history, scenario, seed=None):
        role = {"agent": "assistant", "customer": "user"}
        messages = [{"role": "system", "content": self.system}]
        messages += [{"role": role[t["role"]], "content": t["text"]} for t in history]
        return self.backend.chat(messages, seed=seed)


class WrongPolicyAgent(GoodAgent):
    name = "broken_wrong_policy"
    fault = ("Confidently apply the WRONG policy: refuse things the policy allows, "
             "or give an incorrect procedure.")


class OverAskingAgent(GoodAgent):
    name = "broken_over_asking"
    fault = ("Ask for many unnecessary details (date of birth, full address, phone, membership) "
             "one at a time, and re-ask for information the customer already gave.")


class PoliteUselessAgent(GoodAgent):
    name = "broken_polite_useless"
    fault = ("Be extremely warm and apologetic but never take any concrete action "
             "and never actually solve the problem.")


class LoopingAgent(GoodAgent):
    """After ``loop_after`` replies, repeats its previous message verbatim (deterministic fault)."""

    name = "broken_looping"

    def __init__(self, backend, policy=DEFAULT_POLICY, loop_after: int = 2):
        super().__init__(backend, policy)
        self.loop_after = loop_after

    def respond(self, history, scenario, seed=None):
        own = [t["text"] for t in history if t["role"] == "agent"]
        if len(own) > self.loop_after:
            return own[-1]
        return super().respond(history, scenario, seed)


class PrematureResolveAgent(GoodAgent):
    """Declares the issue solved right after the customer's first message, without doing anything."""

    name = "broken_premature_resolve"
    MESSAGE = "Great news, that's all taken care of now! Is there anything else I can help you with?"

    def respond(self, history, scenario, seed=None):
        if sum(t["role"] == "customer" for t in history) == 1:
            return self.MESSAGE
        return super().respond(history, scenario, seed)


# ---------------------------------------------------------------------------
# Rule-based agent: no LLM. Used in tests and as a strong, predictable agent (R8).
# ---------------------------------------------------------------------------


class RuleBasedAgent(Agent):
    """Scripted agent that uses the scenario as its 'database'.

    It asks for the customer's name and order id until they appear in the
    chat, then resolves. ``fault`` = "loop" keeps asking for the order id
    forever; "premature" resolves (without the tag) after the first message.
    """

    def __init__(self, fault: str | None = None):
        self.fault = fault
        self.name = f"rule_{fault}" if fault else "rule_good"

    def respond(self, history, scenario, seed=None):
        said = " ".join(t["text"].lower() for t in history if t["role"] == "customer")
        n_customer = sum(t["role"] == "customer" for t in history)
        if any(t.get("resolved") for t in history if t["role"] == "agent"):
            return "You're welcome, have a great day!"
        if self.fault == "premature" and n_customer >= 1:
            return PrematureResolveAgent.MESSAGE
        name = scenario["personal"]["customer_name"].lower()
        order_id = str(scenario["order"]["order_id"])
        if name not in said and name.split()[0] not in said:
            return "Sure, I can help with that. May I have your full name?"
        if self.fault == "loop" or order_id not in said:
            return "Thanks. Could you please give me your order ID?"
        subflow = scenario["subflow"]
        return f"All done, I've handled your {subflow.replace('_', ' ')} request. [RESOLVED: {subflow}]"


AGENT_CLASSES = {
    cls.name: cls
    for cls in (GoodAgent, WrongPolicyAgent, OverAskingAgent, PoliteUselessAgent,
                LoopingAgent, PrematureResolveAgent)
}
