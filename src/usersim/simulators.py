"""Customer simulators. Each one reads the scenario + chat so far and returns
the next customer message, or signals the end of the conversation.

The chat is presented exactly as in training (prepare_data.to_messages):
system = scenario, agent = "user", customer = "assistant".
"""

from __future__ import annotations

from .agents import Backend
from .prepare_data import CHAT_START, END_TOKEN, render_scenario

BASIC_STYLE = "Play this customer. Reply with the customer's next message only."

REALISTIC_STYLE = """Play this customer like a real person in a support chat, not an assistant.
- Write short, casual messages; typos and lowercase are fine.
- Do not volunteer all your details at once; answer what you are asked.
- If the agent is unhelpful, repeats itself or gets things wrong, get impatient, push back,
  and leave if it does not improve.
- Never act like the support agent.
Reply with the customer's next message only."""


class Simulator:
    name = "simulator"

    def __init__(self, backend: Backend, style: str = ""):
        self.backend = backend
        self.style = style

    def build_messages(self, scenario: dict, history: list[dict]) -> list[dict]:
        system = render_scenario(scenario) + (f"\n\n{self.style}" if self.style else "")
        role = {"agent": "user", "customer": "assistant"}
        messages = [{"role": role[t["role"]], "content": t["text"]} for t in history]
        if not messages or messages[0]["role"] == "assistant":
            messages.insert(0, {"role": "user", "content": CHAT_START})
        return [{"role": "system", "content": system}] + messages

    def respond(self, scenario: dict, history: list[dict], seed: int | None = None) -> tuple[str, bool]:
        """Return (customer text, ended). The END token is removed from the text."""
        reply = self.backend.chat(self.build_messages(scenario, history), seed=seed)
        ended = END_TOKEN in reply
        return reply.replace(END_TOKEN, "").strip(), ended


class PromptedSimulator(Simulator):
    """Off-the-shelf instruct model + prompt. style: "basic" (B1) or "realistic" (B2)."""

    def __init__(self, backend: Backend, style: str = "basic"):
        super().__init__(backend, {"basic": BASIC_STYLE, "realistic": REALISTIC_STYLE}[style])
        self.name = f"prompted_{style}"


class FineTunedSimulator(Simulator):
    """Our LoRA model. Uses the bare training prompt (no extra style text).

    backend: OllamaBackend("usersim-qwen3-4b") for the GGUF export, or
    HFBackend(base, adapter=...) on Kaggle.
    """

    name = "ft"

    def __init__(self, backend: Backend):
        super().__init__(backend, style="")


class UserLMSimulator(Simulator):
    """Optional baseline B4: microsoft/UserLM-8b.

    TODO: UserLM expects its own system prompt describing the user's intent
    (see its model card); map render_scenario() to that format and verify the
    license before using it.
    """

    name = "userlm"

    def build_messages(self, scenario, history):
        raise NotImplementedError("UserLM prompt format not wired yet")


def scripted(replies: list[str]) -> Simulator:
    """A simulator that plays back fixed replies; useful for tests and dry runs."""

    class _Scripted(Backend):
        def __init__(self):
            self.i = 0

        def chat(self, messages, seed=None):
            reply = replies[min(self.i, len(replies) - 1)]
            self.i += 1
            return reply

    sim = Simulator(_Scripted())
    sim.name = "scripted"
    return sim
