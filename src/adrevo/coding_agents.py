import json
import subprocess
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def uncached_input_tokens(self) -> int:
        return self.input_tokens - self.cached_input_tokens

    def __iadd__(self, other: "TokenUsage"):
        self.input_tokens += other.input_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_output_tokens += other.reasoning_output_tokens
        return self


class Codex:
    def __init__(
        self,
        cwd: str | Path,
        session_id: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ):
        self.cwd = Path(cwd).expanduser().resolve()
        self.session_id = session_id
        self.model = model
        self.reasoning_effort = reasoning_effort

        self.last_events: list[dict] = []
        self.last_usage = TokenUsage()
        self.total_usage = TokenUsage()

    def _config_args(self) -> list[str]:
        args = []

        if self.model:
            args.extend(["--model", self.model])

        if self.reasoning_effort:
            args.extend([
                "--config",
                f"model_reasoning_effort={self.reasoning_effort}",
            ])

        return args

    def _run(self, cmd: list[str], msg: str) -> list[dict]:
        self.last_events = []
        self.last_usage = TokenUsage()

        # Keep stderr separate so stdout remains valid JSONL.
        with tempfile.TemporaryFile(mode="w+t") as stderr:
            process = subprocess.Popen(
                cmd,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                bufsize=1,
            )

            # Write prompt through stdin.
            process.stdin.write(msg)
            process.stdin.close()

            # Stream and collect JSON events.
            for line in process.stdout:
                displayed_line = line.rstrip("\n")
                if len(displayed_line) > 77:
                    displayed_line = f"{displayed_line[:77]}..."
                print(displayed_line, flush=True)

                if not line.strip():
                    continue

                event = json.loads(line)
                self.last_events.append(event)

                if event.get("type") == "thread.started":
                    if self.session_id is None:
                        self.session_id = event["thread_id"]

            returncode = process.wait()

            stderr.seek(0)
            error_output = stderr.read()

        events = self.last_events

        # Track token usage.
        self.last_usage = self.get_usage(events)
        self.total_usage += self.last_usage

        if returncode != 0:
            raise RuntimeError(
                f"Codex exited with code {returncode}:\n{error_output}"
            )

        for event in events:
            if event.get("type") in ("turn.failed", "error"):
                raise RuntimeError(f"Codex execution failed: {event}")

        if not any(e.get("type") == "turn.completed" for e in events):
            raise RuntimeError("Codex execution did not complete")

        return events

    def initial_msg(self, msg: str) -> list[dict]:
        if self.session_id is not None:
            raise RuntimeError("Session already initialized")

        return self._run(
            [
                "codex",
                "exec",
                "--json",
                *self._config_args(),
                "-",
            ],
            msg,
        )

    def send_msg(self, msg: str) -> list[dict]:
        if self.session_id is None:
            raise RuntimeError("Call initial_msg() first")

        return self._run(
            [
                "codex",
                "exec",
                "resume",
                "--json",
                *self._config_args(),
                self.session_id,
                "-",
            ],
            msg,
        )

    @staticmethod
    def get_response(events: list[dict]) -> str:
        messages = [
            event["item"]["text"]
            for event in events
            if event.get("type") == "item.completed"
            and event.get("item", {}).get("type") == "agent_message"
        ]

        return messages[-1] if messages else ""

    @staticmethod
    def get_usage(events: list[dict]) -> TokenUsage:
        usage = TokenUsage()

        for event in events:
            if event.get("type") != "turn.completed":
                continue

            data = event.get("usage", {})

            usage += TokenUsage(
                input_tokens=data.get("input_tokens", 0),
                cached_input_tokens=data.get("cached_input_tokens", 0),
                output_tokens=data.get("output_tokens", 0),
                reasoning_output_tokens=data.get(
                    "reasoning_output_tokens", 0
                ),
            )

        return usage
