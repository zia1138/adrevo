"""Simple streaming wrappers for Codex, Claude Code, and Gemini CLI.

Requires the relevant CLI executable on PATH and already authenticated.
Each wrapper manages one conversation; do not invoke the same instance concurrently.

Terminal output is shortened to PRINT_LIMIT characters plus '...' when longer.
The complete structured JSON events are kept in ``last_events`` and returned.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterator


PRINT_LIMIT = 90


def _shorten(s: str, limit: int = PRINT_LIMIT) -> str:
    """Truncate long printable lines, not the structured data we store."""
    s = s.rstrip("\r\n")
    return s[:limit] + "..." if len(s) > limit else s


@dataclass
class TokenUsage:
    # input_tokens includes cached input tokens for all three wrappers.
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def uncached_input_tokens(self) -> int:
        return max(
            0,
            self.input_tokens - self.cached_input_tokens - self.cache_creation_input_tokens,
        )

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))
        return self

    def __sub__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(**{
            f.name: max(0, getattr(self, f.name) - getattr(other, f.name))
            for f in fields(self)
        })


def _latest_result(events: list[dict]) -> dict | None:
    return next((e for e in reversed(events) if e.get("type") == "result"), None)


class _Agent:
    """Shared subprocess lifecycle; subclasses provide commands and event parsing."""

    def __init__(
        self,
        cwd: str | Path,
        session_id: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.cwd = Path(cwd).expanduser().resolve()
        self.session_id = session_id
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.last_events: list[dict] = []
        self.last_stderr: str = ""
        self.last_usage: TokenUsage | None = TokenUsage()
        self.total_usage = TokenUsage()
        self.last_cost_usd: float | None = None
        self.total_cost_usd: float | None = None

    def initial_msg(self, msg: str) -> list[dict]:
        if self.session_id is not None:
            raise RuntimeError("Session already initialized; use send_msg()")
        return self._invoke(msg, resume=False)

    def send_msg(self, msg: str) -> list[dict]:
        if self.session_id is None:
            raise RuntimeError("Call initial_msg() first")
        return self._invoke(msg, resume=True)

    @contextmanager
    def _environment(self) -> Iterator[dict[str, str]]:
        yield os.environ.copy()

    def _command(self, resume: bool) -> list[str]:
        raise NotImplementedError

    def _accept_event(self, event: dict) -> None:
        raise NotImplementedError

    def _account(self, events: list[dict]) -> None:
        raise NotImplementedError

    def _validate(self, events: list[dict]) -> None:
        raise NotImplementedError

    def _invoke(self, msg: str, resume: bool) -> list[dict]:
        self.last_events = []
        self.last_stderr = ""
        self.last_usage = None
        self.last_cost_usd = None
        bad_lines: list[str] = []
        cmd = self._command(resume)

        # A seekable stdin file avoids deadlocks when writing very large prompts
        # into a process that is also producing JSONL output.
        with self._environment() as env:
            with (
                tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdin,
                tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr,
            ):
                stdin.write(msg)
                stdin.seek(0)

                process = subprocess.Popen(
                    cmd,
                    cwd=self.cwd,
                    env=env,
                    stdin=stdin,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert process.stdout is not None
                try:
                    for line in process.stdout:
                        print(_shorten(line), flush=True)
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                            if not isinstance(event, dict):
                                raise ValueError("JSONL event is not an object")
                        except (json.JSONDecodeError, ValueError):
                            bad_lines.append(_shorten(line))
                            continue
                        self.last_events.append(event)
                        self._accept_event(event)
                finally:
                    process.stdout.close()
                exit_code = process.wait()
                stderr.seek(0)
                self.last_stderr = stderr.read()

        self._account(self.last_events)
        if exit_code:
            raise RuntimeError(
                f"{cmd[0]} exited with code {exit_code}: "
                f"{_shorten(self.last_stderr.strip(), 300)}"
            )
        if bad_lines:
            raise RuntimeError(f"Unexpected non-JSON output from {cmd[0]}: {bad_lines[0]}")
        self._validate(self.last_events)
        return self.last_events


class Codex(_Agent):
    """Codex exec (per-turn JSON usage, no CLI dollar-cost estimate)."""

    def _command(self, resume: bool) -> list[str]:
        config = []
        if self.model:
            config.extend(["--model", self.model])
        if self.reasoning_effort:
            config.extend(["--config", f"model_reasoning_effort={self.reasoning_effort}"])
        if resume:
            return ["codex", "exec", "resume", "--json", *config, self.session_id, "-"]
        return ["codex", "exec", "--json", *config, "-"]

    def _accept_event(self, event: dict) -> None:
        if self.session_id is None and event.get("type") == "thread.started":
            self.session_id = event.get("thread_id")

    @staticmethod
    def get_usage(events: list[dict]) -> TokenUsage:
        usage = TokenUsage()
        for event in events:
            if event.get("type") != "turn.completed":
                continue
            d = event.get("usage") or {}
            usage += TokenUsage(
                input_tokens=d.get("input_tokens", 0),
                cached_input_tokens=d.get("cached_input_tokens", 0),
                output_tokens=d.get("output_tokens", 0),
                reasoning_output_tokens=d.get("reasoning_output_tokens", 0),
            )
        return usage

    @staticmethod
    def get_response(events: list[dict]) -> str:
        messages = [
            e.get("item", {}).get("text", "")
            for e in events
            if e.get("type") == "item.completed"
            and e.get("item", {}).get("type") == "agent_message"
        ]
        return messages[-1] if messages else ""

    def _account(self, events: list[dict]) -> None:
        self.last_usage = self.get_usage(events)
        self.total_usage += self.last_usage

    def _validate(self, events: list[dict]) -> None:
        failures = [e for e in events if e.get("type") in ("turn.failed", "error")]
        if failures:
            raise RuntimeError(f"Codex turn failed: {_shorten(json.dumps(failures[-1]), 300)}")
        if not any(e.get("type") == "turn.completed" for e in events):
            raise RuntimeError("Codex turn did not complete")


class ClaudeCode(_Agent):
    """Claude Code stream-json; recent versions report session-cumulative costs."""

    def __init__(
        self,
        cwd: str | Path,
        session_id: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(cwd, session_id, model, reasoning_effort)
        # An externally restored session has unknown historical baselines.
        self._previous_session_usage: TokenUsage | None = (
            TokenUsage() if session_id is None else None
        )
        self._previous_session_cost: float | None = (
            0.0 if session_id is None else None
        )

    def _command(self, resume: bool) -> list[str]:
        cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose"]
        if self.model:
            cmd.extend(["--model", self.model])
        if self.reasoning_effort:
            cmd.extend(["--effort", self.reasoning_effort])
        if resume:
            cmd.extend(["--resume", self.session_id])
        return cmd

    def _accept_event(self, event: dict) -> None:
        if self.session_id is None:
            self.session_id = event.get("session_id")

    @staticmethod
    def get_usage(events: list[dict]) -> TokenUsage:
        result = _latest_result(events) or {}
        models = result.get("modelUsage") or {}
        if models:
            model_records = list(models.values())
            plain = sum(int(d.get("inputTokens", 0)) for d in model_records)
            cached = sum(int(d.get("cacheReadInputTokens", 0)) for d in model_records)
            created = sum(int(d.get("cacheCreationInputTokens", 0)) for d in model_records)
            output = sum(int(d.get("outputTokens", 0)) for d in model_records)
        else:
            # Fallback is main-agent usage for the *last turn*, not all subagents.
            d = result.get("usage") or {}
            plain = int(d.get("input_tokens", 0))
            cached = int(d.get("cache_read_input_tokens", 0))
            created = int(d.get("cache_creation_input_tokens", 0))
            output = int(d.get("output_tokens", 0))
        return TokenUsage(
            input_tokens=plain + cached + created,
            cached_input_tokens=cached,
            cache_creation_input_tokens=created,
            output_tokens=output,
        )

    @staticmethod
    def get_response(events: list[dict]) -> str:
        return str((_latest_result(events) or {}).get("result") or "")

    def _account(self, events: list[dict]) -> None:
        result = _latest_result(events)
        if result is None:
            return
        usage = self.get_usage(events)
        if result.get("modelUsage"):
            # v2.1.277+: session-wide running totals, including resumed turns.
            self.total_usage = usage
            if self._previous_session_usage is not None:
                self.last_usage = usage - self._previous_session_usage
            self._previous_session_usage = usage
        else:
            # Only per-turn main-agent usage available.
            self.last_usage = usage
            self.total_usage += usage

        if result.get("total_cost_usd") is not None:
            cost = float(result["total_cost_usd"])
            self.total_cost_usd = cost
            if self._previous_session_cost is not None:
                self.last_cost_usd = max(0.0, cost - self._previous_session_cost)
            self._previous_session_cost = cost

    def _validate(self, events: list[dict]) -> None:
        result = _latest_result(events)
        if not result:
            raise RuntimeError("Claude Code did not return a result")
        if result.get("is_error") or result.get("subtype") not in (None, "success"):
            raise RuntimeError(f"Claude Code failed: {_shorten(json.dumps(result), 300)}")


def _deep_merge(target: dict, patch: dict) -> dict:
    """Merge objects without deleting unrelated settings; append customOverrides."""
    for key, value in patch.items():
        if key == "customOverrides" and isinstance(value, list):
            target[key] = list(target.get(key, [])) + value
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


class GeminiCLI(_Agent):
    """Gemini CLI stream-json with per-process Gemini 3 thinking-level override.

    Existing user/project settings are inherited. The temporary system-override
    file is also merged with an existing system settings file, when present.
    Gemini 2.5 thinkingBudget is intentionally not mapped to abstract effort.
    """

    def _command(self, resume: bool) -> list[str]:
        cmd = ["gemini", "--output-format", "stream-json"]
        if self.model:
            cmd.extend(["--model", self.model])
        if resume:
            cmd.extend(["--resume", self.session_id])
        return cmd

    @contextmanager
    def _environment(self) -> Iterator[dict[str, str]]:
        env = os.environ.copy()
        if self.reasoning_effort is None:
            yield env
            return
        if not self.model or not self.model.startswith("gemini-3"):
            raise ValueError("Gemini reasoning_effort requires a Gemini 3 model")
        level = self.reasoning_effort.upper()
        if level not in {"MINIMAL", "LOW", "MEDIUM", "HIGH"}:
            raise ValueError(f"Invalid Gemini reasoning_effort: {self.reasoning_effort}")

        # Preserve an existing system-override file if there is one. Global
        # ~/.gemini/settings.json and project .gemini/settings.json load normally.
        existing = env.get("GEMINI_CLI_SYSTEM_SETTINGS_PATH")
        if existing is None:
            if sys.platform == "darwin":
                existing = "/Library/Application Support/GeminiCli/settings.json"
            elif os.name == "nt":
                existing = os.path.join(
                    env.get("PROGRAMDATA", r"C:\ProgramData"), "gemini-cli", "settings.json"
                )
            else:
                existing = "/etc/gemini-cli/settings.json"

        settings = {}
        path = Path(existing).expanduser()
        if path.is_file():
            settings = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(settings, dict):
                raise ValueError(f"Invalid Gemini system settings at {path}")

        override = {
            "modelConfigs": {
                "customOverrides": [{
                    "match": {"model": self.model},
                    "modelConfig": {
                        "generateContentConfig": {
                            "thinkingConfig": {"thinkingLevel": level}
                        }
                    },
                }]
            }
        }
        _deep_merge(settings, override)
        with tempfile.TemporaryDirectory(prefix="gemini-override-") as tmp:
            filename = Path(tmp) / "settings.json"
            filename.write_text(json.dumps(settings), encoding="utf-8")
            env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = str(filename)
            yield env

    def _accept_event(self, event: dict) -> None:
        if self.session_id is None and event.get("type") == "init":
            self.session_id = event.get("session_id")

    @staticmethod
    def get_response(events: list[dict]) -> str:
        # Gemini emits assistant text as incremental message chunks.
        chunks = [
            e.get("content", "")
            for e in events
            if e.get("type") == "message" and e.get("role") == "assistant"
            and isinstance(e.get("content"), str)
        ]
        if chunks:
            return "".join(chunks)
        return str((_latest_result(events) or {}).get("response") or "")

    @staticmethod
    def get_usage(events: list[dict]) -> TokenUsage:
        result = _latest_result(events) or {}
        stats = result.get("stats") or {}
        return TokenUsage(
            input_tokens=int(stats.get("input_tokens", stats.get("input", 0))),
            cached_input_tokens=int(stats.get("cached", 0)),
            output_tokens=int(stats.get("output_tokens", 0)),
        )

    def _account(self, events: list[dict]) -> None:
        if _latest_result(events):
            self.last_usage = self.get_usage(events)
            self.total_usage += self.last_usage

    def _validate(self, events: list[dict]) -> None:
        result = _latest_result(events)
        if result is None:
            raise RuntimeError("Gemini CLI did not return a result")
        if result.get("status") not in (None, "success"):
            raise RuntimeError(f"Gemini CLI failed: {_shorten(json.dumps(result), 300)}")


__all__ = ["Codex", "ClaudeCode", "GeminiCLI", "TokenUsage", "PRINT_LIMIT"]
