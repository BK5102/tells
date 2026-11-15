"""The language-model interface, plus cassettes so tests are deterministic and free.

Everything the agent sends to a model goes through `LLM.create`, which returns
a plain dict: {"stop_reason": ..., "content": [block dicts]}. Three backends:

- `AnthropicLLM`: Claude through the official SDK (needs the `agent` extra and
  credentials: ANTHROPIC_API_KEY or an `ant auth login` profile).
- `CassetteLLM` in "replay" mode: returns recorded responses in order and
  checks each request matches the recording, so a changed prompt fails loudly.
- `CassetteLLM` in "record" mode: wraps a real backend and saves every exchange.

Model choice and settings are in methodology M40.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

MODEL = "claude-opus-5-5"
EFFORT = "high"  # planning experiments is intelligence-sensitive; Opus 5.5 defaults to medium


class AgentRefusedError(RuntimeError):
    """The model declined the request (stop_reason "refusal")."""


class LLM(Protocol):
    def create(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]: ...


def request_digest(system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"system": system, "messages": messages, "tools": tools}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class AnthropicLLM:
    """Claude Opus 5.5 with strict tool schemas and server-side refusal fallback."""

    def __init__(self, model: str = MODEL, effort: str = EFFORT, max_tokens: int = 16000):
        import anthropic

        self._client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def create(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=[{**t, "strict": True} for t in tools],
            output_config={"effort": self.effort},
            # If Opus declines, the API re-runs the request on a fallback model
            # chosen by refusal category, inside the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        out = response.to_dict()
        if out.get("stop_reason") == "refusal":
            raise AgentRefusedError(str(out.get("stop_details")))
        return {"stop_reason": out["stop_reason"], "content": out["content"], "model": out["model"]}


class CassetteLLM:
    """Replay recorded exchanges (default) or record new ones through a real backend."""

    def __init__(self, path: str | Path, mode: str = "replay", backend: LLM | None = None):
        if mode not in ("replay", "record"):
            raise ValueError("mode must be 'replay' or 'record'")
        if mode == "record" and backend is None:
            raise ValueError("record mode needs a real backend")
        self.path = Path(path)
        self.mode = mode
        self.backend = backend
        self._recorded: list[dict[str, Any]] = (
            json.loads(self.path.read_text(encoding="utf-8")) if mode == "replay" else []
        )
        self._i = 0

    def create(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        digest = request_digest(system, messages, tools)
        if self.mode == "record":
            assert self.backend is not None
            response = self.backend.create(system=system, messages=messages, tools=tools)
            self._recorded.append({"request_digest": digest, "response": response})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._recorded, indent=2), encoding="utf-8")
            return response
        if self._i >= len(self._recorded):
            raise RuntimeError(f"cassette {self.path.name} has no response #{self._i + 1}")
        entry = self._recorded[self._i]
        self._i += 1
        expected = entry.get("request_digest")
        if expected not in (None, "any") and expected != digest:
            raise RuntimeError(
                f"request #{self._i} differs from the recording in {self.path.name}; "
                "re-record the cassette if the change is intended"
            )
        return entry["response"]
