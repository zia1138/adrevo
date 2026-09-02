import logging
import random
import textwrap
import time
import uuid
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any

import logfire
import ray
from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage

from adrevo.config import AdrevoConfig, BackendConfig, ModelSpec
from adrevo.database import Program, ProgramDatabase
from adrevo.ray_backend import RayExecutionBackend
from adrevo.utils import (
    extract_file_to_string,
    extract_file_replacements,
)

logger = logging.getLogger(__name__)
_logging_configured = False


class ModelResponseKind(Enum):
    # Expected shape/purpose of the selected LLM's next response. A "turn" is the
    # LLM call itself; this enum says how to parse and handle that call's output.
    # CODE_UPDATE -> replacement code to evaluate.
    # DIAGNOSE -> diagnostic code to run, then feed back into CODE_UPDATE.
    CODE_UPDATE = "code_update"
    DIAGNOSE = "diagnose"


@dataclass
class ScheduledTurn:
    """The next LLM call to run and how its response should be interpreted."""

    prompt: str
    expected_response: ModelResponseKind


class ModelOutcome(Enum):
    # Final disposition for one model session on the claimed parent.
    IMPROVED = "improved"
    FAILED = "failed"
    STALE = "stale"
    STOPPED = "stopped"

    @property
    def should_stop_claim(self) -> bool:
        return self in {ModelOutcome.IMPROVED, ModelOutcome.STALE, ModelOutcome.STOPPED}


class StaleClaimError(Exception):
    """Raised to exit the current parent claim when the DB makes it obsolete."""


class CostLimitExceededError(Exception):
    """Raised to exit the current model when the run reaches its cost limit."""


@dataclass
class ParentRunContext:
    """Mutable state for one worker run against one claimed parent program."""

    # Generation and cost snapshot at the time this parent run started.
    current_gen: int
    cur_cost: float
    # Parent program selected from the DB, plus its zipped project state.
    claim_id: str
    parent: Program
    parent_zip_bytes: bytes
    # Evaluation file contents included in the initial model prompt.
    eval_code: str
    # Project zip used for evaluation.
    current_zip_bytes: bytes
    # Cached parent score and start time used for feedback and DB metadata.
    parent_score: float
    inference_start: float
    # Optional strategy sampled for this parent run and appended to prompts.
    strategy: str | None
    improved_child_id: str | None = None


@dataclass
class ModelSession:
    """One selected LLM's multi-turn attempt to improve the claimed parent.

    The session always has a scheduled_turn. Running that turn sends its prompt
    to the LLM and parses the output according to its expected_response. Handling
    the response either ends the session or schedules a replacement turn.
    """

    # The coordinator already picked this model and rank for the parent run.
    # Everything below belongs to this one conversation with that selected LLM.
    selected_model: ModelSpec
    model_rank: int
    agent: Agent
    # The next LLM call waiting to be run inside this session.
    scheduled_turn: ScheduledTurn
    # Full conversation history returned by pydantic-ai. This is what keeps every
    # LLM turn in this selected model session connected to earlier feedback.
    message_history: Any = None
    # Counts failed code-update evaluations since the last diagnostic detour.
    consecutive_code_update_failures: int = 0
    # Complete replacements extracted from the response that just came back.
    turn_payload: dict[str, str] | None = None
    # Evaluation feedback from the accepted code update, kept so an optional
    # final summary turn can ask the same LLM what strategy worked.
    success_feedback_prompt: str = ""
    # Latest evaluation feedback from any code update. If the session ends after
    # the final allowed model turn, this is the only place that feedback lives.
    last_evaluation_feedback: str = ""


@ray.remote(num_cpus=0)
class AdrevoState:
    """Shared state for the Adrevo algorithm, stored in a Ray actor.
       Separate from the ProgramDatabase to shard state a bit."""
    def __init__(
        self,
        config: AdrevoConfig,
        initial: int = 0,
        checkpoint: dict[str, Any] | None = None,
    ):
        self.config = config
        self.evo_models = config.build_evo_models()
        self.model_costs = {
            spec.model_id: (spec.input_token_cost, spec.output_token_cost)
            for spec in self.evo_models
        }
        if checkpoint is None:
            self.generation = initial
            self.carried_cost = 0.0
            self.input_tokens = {model_id: 0 for model_id in self.model_costs}
            self.output_tokens = {model_id: 0 for model_id in self.model_costs}
        else:
            self._restore_checkpoint(checkpoint)

    def _restore_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.generation = checkpoint["generation"]
        self.carried_cost = checkpoint["carried_cost"] + self._checkpoint_usage_cost(
            checkpoint
        )

        # A resumed run starts a new accounting segment using its current model
        # configuration. Historical model usage is represented by carried_cost.
        self.input_tokens = {model_id: 0 for model_id in self.model_costs}
        self.output_tokens = {model_id: 0 for model_id in self.model_costs}

    def _checkpoint_usage_cost(
        self,
        checkpoint: dict[str, Any],
    ) -> float:
        saved_model_costs = checkpoint["model_costs"]
        saved_input_tokens = checkpoint["input_tokens"]
        saved_output_tokens = checkpoint["output_tokens"]

        total_cost = 0.0
        for model_id, costs in saved_model_costs.items():
            input_token_cost, output_token_cost = costs
            total_cost += (
                float(saved_input_tokens[model_id] / 1e6) * input_token_cost
            )
            total_cost += (
                float(saved_output_tokens[model_id] / 1e6) * output_token_cost
            )
        return total_cost

    def update_usage(self, model_id: str, usage: RunUsage):
        if model_id not in self.model_costs:
            raise ValueError(f"Unknown model_id for usage accounting: {model_id}")
        self.input_tokens[model_id] += usage.input_tokens
        self.output_tokens[model_id] += usage.output_tokens
        # Count reasoning tokens as output tokens.
        if "reasoning_tokens" in usage.details:
            self.output_tokens[model_id] += usage.details["reasoning_tokens"]

    def get_gen(self):
        return self.generation

    def next_gen(self):
        self.generation += 1
        return self.generation

    def snapshot(self) -> dict[str, Any]:
        """Return one actor-consistent checkpoint for generation and cost state."""
        return {
            "generation": self.generation,
            "carried_cost": self.carried_cost,
            "model_costs": {
                model_id: list(costs)
                for model_id, costs in self.model_costs.items()
            },
            "input_tokens": dict(self.input_tokens),
            "output_tokens": dict(self.output_tokens),
        }

    def compute_cost(self):
        total_cost = self.carried_cost
        for model_id, (input_token_cost, output_token_cost) in self.model_costs.items():
            total_cost += (float(self.input_tokens[model_id] / 1e6) * input_token_cost)
            total_cost += (float(self.output_tokens[model_id] / 1e6) * output_token_cost)
        return total_cost


@ray.remote(num_cpus=0)
class AdrevoModelCoordinator:
    """Coordinates finite concurrent model leases across all workers.

    Model ranks form an escalation ladder: rank 0 is the cheapest/default model,
    and higher ranks are increasingly capable, expensive, and usually have fewer
    concurrent leases. Workers try rank 0 first and escalate only after failure.
    """

    def __init__(self, config: AdrevoConfig):
        # Each ladder entry records one escalation tier and its concurrency cap.
        self.model_ladder = [
            {
                "rank": rank,
                "model_id": spec.model_id,
                "max_concurrent_leases": spec.max_concurrent_leases,
            }
            for rank, spec in enumerate(config.build_evo_models())
        ]
        # Remaining slots per model. Higher-rank models are typically scarce, so
        # leasing prevents many workers from piling onto the expensive tiers.
        self.available_leases = {
            item["model_id"]: item["max_concurrent_leases"]
            for item in self.model_ladder
        }
        # One waiting line per rank. A worker waiting for an expensive rank should
        # not block another worker that can still use a cheaper rank.
        self.pending_queues = {
            item["rank"]: deque() for item in self.model_ladder
        }
        # Queue entries are request ids; this stores the details for each id.
        self.pending_requests: dict[str, dict[str, Any]] = {}
        # Leases that have been handed out and not yet released.
        self.active_leases: dict[str, dict[str, Any]] = {}

    def acquire_model(
        self,
        worker_id: str,
        parent_id: str,
        generation: int,
        rank: int = 0,
    ) -> dict[str, Any] | None:
        # Immediate path: use this rank now if there is capacity and no existing
        # waiter for the same rank. This keeps scarce higher-rank leases fair.
        if self.pending_queues[rank]:
            return None
        lease = self._reserve_available_lease(worker_id, parent_id, generation, rank)
        if lease is None:
            return None
        self._activate_lease(lease)
        return lease

    def enqueue_model_request(
        self,
        worker_id: str,
        parent_id: str,
        generation: int,
        rank: int = 0,
    ) -> str:
        # Waiting path: join the line for this exact rank. The worker does not
        # fall back to a lower rank while waiting; it either gets this tier or
        # abandons the parent if the work becomes stale.
        request_id = str(uuid.uuid4())
        self.pending_requests[request_id] = {
            "request_id": request_id,
            "worker_id": worker_id,
            "parent_id": parent_id,
            "generation": generation,
            "rank": rank,
        }
        self.pending_queues[rank].append(request_id)
        return request_id

    def poll_model_request(self, request_id: str) -> dict[str, Any] | None:
        request = self.pending_requests.get(request_id)
        if request is None:
            return None
        rank = request["rank"]
        rank_queue = self.pending_queues[rank]
        # Only the front waiter for this rank can claim the next scarce slot.
        if not rank_queue or rank_queue[0] != request_id:
            return None

        lease = self._reserve_available_lease(
            request["worker_id"],
            request["parent_id"],
            request["generation"],
            rank,
        )
        if lease is None:
            return None
        rank_queue.popleft()
        del self.pending_requests[request_id]
        self._activate_lease(lease)
        return lease

    def cancel_model_request(self, request_id: str) -> None:
        # A waiting worker can give up if its parent becomes stale or cost stops.
        request = self.pending_requests.pop(request_id, None)
        if request is None:
            return
        rank = request["rank"]
        self.pending_queues[rank] = deque(
            queued_request_id
            for queued_request_id in self.pending_queues[rank]
            if queued_request_id != request_id
        )

    def release_model(self, lease_id: str) -> None:
        # Work is done: return the model slot so the next waiter can claim it.
        lease = self.active_leases.pop(lease_id, None)
        if lease is None:
            return
        self._return_lease(lease)

    def _reserve_available_lease(
        self,
        worker_id: str,
        parent_id: str,
        generation: int,
        rank: int,
    ) -> dict[str, Any] | None:
        # Reserve capacity for this rank's model. The caller will either activate
        # the lease immediately or use it to promote the front waiter.
        item = self.model_ladder[rank]
        model_id = item["model_id"]
        if self.available_leases[model_id] <= 0:
            return None

        self.available_leases[model_id] -= 1
        return {
            "lease_id": str(uuid.uuid4()),
            "model_id": model_id,
            "rank": rank,
            "worker_id": worker_id,
            "parent_id": parent_id,
            "generation": generation,
        }

    def _activate_lease(self, lease: dict[str, Any]) -> None:
        self.active_leases[lease["lease_id"]] = lease

    def _return_lease(self, lease: dict[str, Any]) -> None:
        model_id = lease["model_id"]
        self.available_leases[model_id] += 1


@ray.remote(num_cpus=1)
class AdrevoWorker:
    """
    adrevo worker

    NOTE: Each worker reserves one CPU because it runs local evaluation
    subprocesses through RayExecutionBackend.
    """
    def __init__(
        self,
        worker_id: str,
        state: AdrevoState,
        evo_config: AdrevoConfig,
        backend_config: BackendConfig,
        db: ProgramDatabase,
        model_coordinator: AdrevoModelCoordinator,
        verbose: bool,
        data_handle=None,
    ):
        super().__init__()
        self.worker_id = worker_id
        self.state = state
        self.evo_config = evo_config
        self.db = db
        self.model_coordinator = model_coordinator
        self.verbose = verbose

        self.backend = RayExecutionBackend(
            config=backend_config,
            evaluator_file=evo_config.evaluate_file,
            verbose=verbose,
            data_handle=data_handle,
        )

        global _logging_configured
        if self.verbose:
            if not _logging_configured:
                logfire.configure(scrubbing=False)
                logger.addHandler(logfire.LogfireLoggingHandler())
                logfire.instrument_pydantic_ai()
                _logging_configured = True
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)

        # Build models once within the actor.
        self.evo_models = self.evo_config.build_evo_models()
        self.model_by_id = {spec.model_id: spec for spec in self.evo_models}

    def max_cost_exceeded(self) -> bool:
        cur_cost: float = ray.get(self.state.compute_cost.remote())
        if cur_cost >= self.evo_config.max_cost:
            logger.info(
                f"Worker {self.worker_id}: Current cost ({cur_cost}) exceeds max cost "
                f"{self.evo_config.max_cost}. Stopping evolution."
            )
            return True
        return False

    def cancellation_requested(self) -> bool:
        if ray.get(self.db.is_cancelled.remote()):
            logger.info(
                f"Worker {self.worker_id}: Database cancellation requested. "
                "Stopping worker."
            )
            return True
        return False

    def run(self):
        """Main agent loop for the worker."""
        with self.backend:
            cur_cost: float = ray.get(self.state.compute_cost.remote())
            while True:
                if self.cancellation_requested():
                    break

                current_gen: int = ray.get(self.state.next_gen.remote())
                self.agent_run(current_gen, cur_cost)

                if self.cancellation_requested():
                    break

                if current_gen >= self.evo_config.max_generations - 1:
                    logger.info(
                        f"Worker {self.worker_id}: Reached max generations "
                        f"({self.evo_config.max_generations}). Stopping evolution."
                    )
                    break

                cur_cost = ray.get(self.state.compute_cost.remote())
                if cur_cost >= self.evo_config.max_cost:
                    logger.info(
                        f"Worker {self.worker_id}: Current cost ({cur_cost}) exceeds "
                        f"max cost {self.evo_config.max_cost}. Stopping evolution."
                    )
                    break

    def agent_run(self, current_gen: int, cur_cost: float):
        """Run one generation against one claimed parent program."""
        context = self._sample_parent_context(current_gen, cur_cost)
        if context is None:
            return

        claim_outcome = ModelOutcome.FAILED
        next_rank = 0
        try:
            while next_rank < len(self.evo_models):
                lease = self._acquire_or_wait_for_model(context, next_rank)
                if lease is None:
                    break

                selected_model = self.model_by_id[lease["model_id"]]
                ray.get(
                    self.db.update_claim_model.remote(
                        context.claim_id,
                        selected_model.model_id
                    )
                )
                logger.info(
                    f"Worker {self.worker_id}: Using model {selected_model.model_id} "
                    f"(rank {lease['rank']}) for generation {current_gen}."
                )
                try:
                    claim_outcome = self._try_improve_with_model(
                        context,
                        selected_model,
                        lease["rank"],
                    )
                finally:
                    self._release_model_lease(lease)

                if claim_outcome.should_stop_claim:
                    return
                next_rank = lease["rank"] + 1

            if next_rank >= len(self.evo_models):
                logger.info(f"Worker {self.worker_id}: All model ranks exhausted for this parent run. No improvement found.")

            self._raise_if_claim_stale(
                context,
                "Claim became stale after model ranks were exhausted.",
            )
        except StaleClaimError:
            claim_outcome = ModelOutcome.STALE
        except CostLimitExceededError:
            claim_outcome = ModelOutcome.STOPPED
        finally:
            ray.get(
                self.db.finish_parent_claim.remote(
                    context.claim_id,
                    claim_outcome.value,
                    context.improved_child_id,
                )
            )

    def _sample_parent_context(
        self,
        current_gen: int,
        cur_cost: float,
    ) -> ParentRunContext | None:
        logger.info(
            f"Worker {self.worker_id} gen={current_gen}; cur_cost: {cur_cost:.4f}"
        )
        claim = ray.get(self.db.claim_next_parent.remote(self.worker_id, current_gen))
        if claim is None:
            logger.info(
                f"Worker {self.worker_id} at Generation {current_gen}: "
                "No parent claim available. Skipping."
            )
            return None

        parent = claim["parent"]
        if parent is None:
            logger.info(
                f"Worker {self.worker_id} at Generation {current_gen}: "
                "No parent found. Skipping."
            )
            return None

        parent_zip_bytes: bytes = ray.get(self.db.get_zip_bytes.remote(parent.id))
        strategy = self._select_strategy(current_gen)
        eval_code = extract_file_to_string(
            parent_zip_bytes,
            self.evo_config.evaluate_file,
        )

        return ParentRunContext(
            current_gen=current_gen,
            cur_cost=cur_cost,
            claim_id=claim["claim_id"],
            parent=parent,
            parent_zip_bytes=parent_zip_bytes,
            eval_code=eval_code,
            current_zip_bytes=parent_zip_bytes,
            parent_score=parent.combined_score,
            inference_start=time.time(),
            strategy=strategy,
        )

    def _select_strategy(self, current_gen: int) -> str | None:
        strategy = random.choices(
            (None, *self.evo_config.strategies),
            weights=(self.evo_config.pr_no_strategy, *self.evo_config.pr_strategies),
            k=1,
        )[0]
        if strategy:
            logger.info(
                f"Worker {self.worker_id} at Generation {current_gen}: "
                f"Injecting strategy into prompt: {strategy}"
            )
        else:
            logger.info(
                f"Worker {self.worker_id} at Generation {current_gen}: "
                "No strategy injected for this parent run."
            )
        return strategy

    def _try_improve_with_model(
        self,
        context: ParentRunContext,
        selected_model: ModelSpec,
        model_rank: int,
    ) -> ModelOutcome:
        context.current_zip_bytes = context.parent_zip_bytes

        session = self._start_model_session(context, selected_model, model_rank)

        try:
            for turn_idx in range(session.selected_model.max_model_turns):
                turns_remaining = (
                    session.selected_model.max_model_turns - turn_idx - 1
                )
                # One loop iteration spends one LLM turn. At the start of the
                # turn, session.scheduled_turn is the contract: what we ask for
                # and how the reply will be interpreted.
                response = self._run_model_turn(
                    context,
                    session,
                )
                if response is None:
                    continue

                try:
                    # Convert the LLM's fenced reply into the raw submission for
                    # this turn. The response kind decides which fence/parser is
                    # valid; a malformed reply schedules a repair prompt instead.
                    session.turn_payload = None
                    session.turn_payload = self._extract_turn_payload(response.output)
                except ValueError as exc:
                    session.scheduled_turn = ScheduledTurn(
                        prompt=self._format_missing_fence_prompt(
                            context,
                            session,
                            exc,
                        ),
                        expected_response=session.scheduled_turn.expected_response,
                    )
                    continue

                outcome = self._handle_model_response(
                    context,
                    session,
                    turns_remaining,
                )
                # Handling the response either ends the session or schedules the
                # next turn by replacing session.scheduled_turn.
                if outcome == ModelOutcome.IMPROVED:
                    return self._summarize_success_if_needed(
                        session,
                    )
                if outcome is not None:
                    return outcome

            # Terminal transition after max_model_turns without improvement:
            # if the selected LLM never produced an improved code update, this
            # model session failed and the parent run may escalate to a higher rank.
            if self.verbose and session.message_history is not None:
                if self.max_cost_exceeded():
                    return ModelOutcome.STOPPED
                # This just adds to the logfire output/logs.
                try:
                    response = session.agent.run_sync(
                        self._format_failure_summary_prompt(session),
                        message_history=session.message_history,
                    )
                    ray.get(
                        self.state.update_usage.remote(
                            session.selected_model.model_id,
                            response.usage,
                        )
                    )
                except Exception as exc:
                    logger.error(
                        f"Worker {self.worker_id} at Generation {context.current_gen}: "
                        f"Error during LLM inference for failure summary: {session.selected_model.model_id}: {exc}"
                    )
        except StaleClaimError:
            return ModelOutcome.STALE
        except CostLimitExceededError:
            return ModelOutcome.STOPPED

        return ModelOutcome.FAILED

    def _start_model_session(
        self,
        context: ParentRunContext,
        selected_model: ModelSpec,
        model_rank: int,
    ) -> ModelSession:
        # Build the pydantic-ai Agent once for this selected LLM. The session then
        # carries that agent and its message history through all follow-up turns.
        agent = Agent(
            selected_model.model,
            system_prompt=self.evo_config.task_sys_msg,
            model_settings=selected_model.settings,
        )
        return ModelSession(
            selected_model=selected_model,
            model_rank=model_rank,
            agent=agent,
            scheduled_turn=ScheduledTurn(
                prompt=self._format_initial_code_update_prompt(
                    context,
                    selected_model.max_model_turns,
                ),
                expected_response=ModelResponseKind.CODE_UPDATE,
            ),
        )

    def _run_model_turn(
        self,
        context: ParentRunContext,
        session: ModelSession,
    ) -> Any | None:

        # Before spending another LLM call, abandon this session if the DB
        # search scheduler has invalidated this claim.
        self._raise_if_claim_stale(
            context,
            "Claim {claim_id} became stale before another LLM turn.",
        )
        try:
            with logfire.suppress_instrumentation():
                if self.max_cost_exceeded():
                    raise CostLimitExceededError
                # First turn starts the conversation. Later turns include the
                # same message history so the LLM can use prior feedback.
                if session.message_history is None:
                    response = session.agent.run_sync(session.scheduled_turn.prompt)
                else:
                    response = session.agent.run_sync(
                        session.scheduled_turn.prompt,
                        message_history=session.message_history,
                    )
                ray.get(
                    self.state.update_usage.remote(
                        session.selected_model.model_id,
                        response.usage,
                    )
                )
                # The turn is now part of the session conversation. Even if the
                # response is malformed, the repair prompt should follow it.
                session.message_history = response.all_messages()
                return response
        except CostLimitExceededError:
            raise
        except Exception as exc:
            logger.error(
                f"Worker {self.worker_id} at Generation {context.current_gen}: "
                f"Error during LLM inference with {session.selected_model.model_id}: {exc}"
            )
            return None

    def _handle_model_response(
        self,
        context: ParentRunContext,
        session: ModelSession,
        turns_remaining: int,
    ) -> ModelOutcome | None:
        if session.turn_payload is None:
            raise RuntimeError("_handle_model_response called without turn_payload")

        # A diagnostic response is a detour inside the same LLM session. The
        # model writes instrumented source, we run it, then its output becomes
        # the next CODE_UPDATE prompt.
        if session.scheduled_turn.expected_response == ModelResponseKind.DIAGNOSE:
            session.scheduled_turn = ScheduledTurn(
                prompt=self._run_diagnostic(
                    context,
                    session.turn_payload,
                ),
                expected_response=ModelResponseKind.CODE_UPDATE,
            )
            session.consecutive_code_update_failures = 0
            return None

        # Code-update responses are the only turns that can produce a program
        # eligible for evaluation and database insertion.
        improved, feedback_prompt = self._evaluate_code_update(
            context,
            session.turn_payload,
            session,
        )
        session.last_evaluation_feedback = feedback_prompt
        if improved:
            session.success_feedback_prompt = feedback_prompt
            return ModelOutcome.IMPROVED

        # A failed code update stays in the same LLM conversation. Scheduling the
        # next turn chooses whether the same LLM should revise directly or first
        # produce diagnostic information.
        session.consecutive_code_update_failures += 1
        self._schedule_next_turn(
            context,
            feedback_prompt,
            session,
            turns_remaining,
        )
        return None

    def _schedule_next_turn(
        self,
        context: ParentRunContext,
        feedback_prompt: str,
        session: ModelSession,
        turns_remaining: int,
    ) -> None:
        if (
            session.consecutive_code_update_failures
            < self.evo_config.code_update_failures_before_diagnostics
        ):
            # Keep the next turn simple: give evaluation feedback to the same LLM
            # and ask it for another code update.
            session.scheduled_turn = ScheduledTurn(
                prompt=self._format_code_update_prompt(
                    context,
                    feedback_prompt,
                    include_revision_instruction=True,
                ),
                expected_response=ModelResponseKind.CODE_UPDATE,
            )
            return

        # A detour only helps if there is still one turn for the detour itself
        # and another turn afterward for the final code update.
        can_take_detour = turns_remaining >= 2

        if can_take_detour and self.evo_config.use_probe:
            # Otherwise use the next turn to gather evidence. The diagnostic run
            # output is fed back into the same session before the next code update.
            session.scheduled_turn = ScheduledTurn(
                prompt=self._format_diagnostic_prompt(
                    context,
                    feedback_prompt,
                ),
                expected_response=ModelResponseKind.DIAGNOSE,
            )
            session.consecutive_code_update_failures = 0
            return

        # If detours are disabled or not selected, keep using direct evaluation
        # feedback as the next prompt in the same LLM session.
        session.scheduled_turn = ScheduledTurn(
            prompt=self._format_code_update_prompt(
                context,
                feedback_prompt,
                include_revision_instruction=True,
            ),
            expected_response=ModelResponseKind.CODE_UPDATE,
        )

    def _evaluate_code_update(
        self,
        context: ParentRunContext,
        file_replacements: dict[str, str],
        session: ModelSession,
    ) -> tuple[bool, str]:
        """Evaluate generated replacements and return ``(improved, feedback_prompt)``.

        ``improved`` is true only when the replacement set beats the claimed parent.
        ``feedback_prompt`` summarizes the evaluation result without deciding
        what the next LLM turn should produce.
        """
        candidate_files = dict(context.parent.files)
        candidate_files.update(file_replacements)
        results, runtime_sec, result_zip_bytes = self.backend.run_job(
            parent_zip_bytes=context.current_zip_bytes,
            file_replacements=file_replacements,
            preempt_db=self.db,
            preempt_claim_id=context.claim_id,
        )
        self._raise_if_result_preempted_or_stale(context, results, "Evaluation")
        if not results:
            return (
                False,
                "The program failed to run. No results were returned.",
            )

        fdback: list[str] = []
        if results.get("correct"):
            fdback.append(
                "The program executed correctly and produced a valid result."
            )
            combined = results.get("combined_score")
            if combined is not None:
                fdback.append(f"It achieved a score of {combined}.")
                db_program = Program(
                    id=str(uuid.uuid4()),
                    files=candidate_files,
                    parent_id=context.parent.id,
                    generation=context.current_gen,
                    model_id=session.selected_model.model_id,
                    correct=True,
                    combined_score=combined,
                    inference_time=time.time() - context.inference_start,
                    compute_time=runtime_sec,
                )

                if combined > context.parent_score:
                    fdback.append(
                        "This is an improvement over the parent score of "
                        f"{context.parent_score}."
                    )
                    context.improved_child_id = ray.get(
                        self.db.add.remote(db_program, result_zip_bytes)
                    )
                    return True, "\n".join(fdback)

                fdback.append(
                    "However, this is not an improvement over the parent score of "
                    f"{context.parent_score}."
                )

                # Keep all programs that execute correctly (if requested?, add flag?)
                ray.get(self.db.add.remote(db_program, result_zip_bytes))
            else:
                fdback.append(
                    "Something happened and the score was not available in results."
                )
        else:
            fdback.append(
                "The program did not execute correctly and did not produce a valid "
                "result."
            )
            fdback.append(
                "Here is the error: `"
                + (results.get("error") or "Unknown Error")
                + "`"
            )

        fdback.append(f"The evaluation took {runtime_sec:.2f} seconds.")
        self._append_captured_output(fdback, results, "program")
        return False, "\n".join(fdback)

    def _run_diagnostic(
        self,
        context: ParentRunContext,
        file_replacements: dict[str, str],
    ) -> str:
        results, _, _ = self.backend.run_job(
            parent_zip_bytes=context.current_zip_bytes,
            file_replacements=file_replacements,
            preempt_db=self.db,
            preempt_claim_id=context.claim_id,
        )
        self._raise_if_result_preempted_or_stale(context, results, "Diagnostic")

        lines: list[str] = []
        self._append_captured_output(lines, results, "diagnostic run")
        if not lines:
            lines.append("The diagnostic code did not produce any output.")

        diagnostic_output = "\n".join(lines)
        return self._format_code_update_prompt(
            context,
            textwrap.dedent("""
                {diagnostic_output}
                Learn from the diagnostic output above and write complete replacements for any candidate files that need to change to improve on the parent score of {parent_score}.
            """).format(
                diagnostic_output=diagnostic_output,
                parent_score=context.parent_score,
            ).strip(),
            include_strategy=False,
        )

    def _acquire_or_wait_for_model(
        self,
        context: ParentRunContext,
        rank: int,
    ) -> dict[str, Any] | None:

        # No more models to try, done.
        # TODO: Need to use this to implement a backtracking signal.
        if rank >= len(self.evo_models):
            return None

        lease = ray.get(
            self.model_coordinator.acquire_model.remote(
                self.worker_id,
                context.parent.id,
                context.current_gen,
                rank,
            )
        )
        if lease is not None:
            return lease

        request_id = ray.get(
            self.model_coordinator.enqueue_model_request.remote(
                self.worker_id,
                context.parent.id,
                context.current_gen,
                rank,
            )
        )
        while True:
            try:
                # No need to wait any more for a stronger model if the DB search
                # scheduler has invalidated this claim.
                self._raise_if_claim_stale(
                    context,
                    "Claim {claim_id} became stale; canceling queued model request.",
                )
            except StaleClaimError:
                ray.get(self.model_coordinator.cancel_model_request.remote(request_id))
                raise
            if self.max_cost_exceeded():
                # We've spent too much, stop.
                ray.get(self.model_coordinator.cancel_model_request.remote(request_id))
                raise CostLimitExceededError

            # Is the model of current rank available?
            lease = ray.get(
                self.model_coordinator.poll_model_request.remote(request_id)
            )
            if lease is not None:
                return lease

            time.sleep(self.evo_config.model_wait_poll_sec)

    def _release_model_lease(self, lease: dict[str, Any]) -> None:
        ray.get(self.model_coordinator.release_model.remote(lease["lease_id"]))

    def _summarize_success_if_needed(
        self,
        session: ModelSession,
    ) -> ModelOutcome:
        if not self.verbose:
            return ModelOutcome.IMPROVED

        if self.max_cost_exceeded():
            return ModelOutcome.STOPPED

        try:
            response = session.agent.run_sync(
                self._format_success_summary_prompt(session),
                message_history=session.message_history,
            )
            ray.get(
                self.state.update_usage.remote(
                    session.selected_model.model_id,
                    response.usage,
                )
            )
        except Exception as exc:
            logger.error(
                f"Worker {self.worker_id} at Generation {session.model_rank}: "
                f"Error during LLM inference for success summary: {exc}"
            )
        return ModelOutcome.IMPROVED

    def _format_initial_code_update_prompt(
        self,
        context: ParentRunContext,
        max_model_turns: int,
    ) -> str:
        parent_files = "\n\n".join(
            textwrap.dedent("""\
                ### {file}
                ```{lang_identifier}
                {contents}
                ```
            """).format(
                file=spec.file,
                lang_identifier=spec.lang_identifier,
                contents=context.parent.files[spec.file],
            ).rstrip()
            for spec in self.evo_config.evolvable_files
        )
        prompt = textwrap.dedent("""
            Improve one or more source code files to achieve a higher `combined_score` than the parent program.
            The parent program score is {score}.
            Improve the current algorithm, replace it with a new algorithm, or install and use new dependencies.

            The current candidate files are:

            {parent_files}

            `{evaluate_file}` builds and evaluates the candidate and computes `combined_score`

            ### {evaluate_file}
            ```python
            {eval_code}
            ```

            Preserve the existing input, output, and entrypoint behavior.
            You have at most {max_model_turns} model turns total, including this one.
            Learn from feedback and revise accordingly.
        """).format(
            score=context.parent.combined_score,
            parent_files=parent_files,
            max_model_turns=max_model_turns,
            evaluate_file=self.evo_config.evaluate_file,
            eval_code=context.eval_code,
        ).strip()
        return self._format_code_update_prompt(context, prompt)

    def _format_code_update_prompt(
        self,
        context: ParentRunContext,
        prompt: str,
        *,
        include_strategy: bool = True,
        include_revision_instruction: bool = False,
    ) -> str:
        prompt = prompt.rstrip()
        if include_revision_instruction:
            prompt += textwrap.dedent("""

                Learn from the feedback above and write complete replacements for any candidate files that need to change.
            """).rstrip()

        if include_strategy and context.strategy:
            prompt += textwrap.dedent("""
                {strategy}
            """).format(strategy=context.strategy).rstrip()

        response_instruction = self._format_file_replacement_contract()

        return prompt + "\n\n" + response_instruction

    def _format_file_replacement_contract(self) -> str:
        allowed_files = "\n".join(
            f"- `{spec.file}` with a ```{spec.lang_identifier}``` code fence"
            for spec in self.evo_config.evolvable_files
        )
        return textwrap.dedent("""
            Return one or more complete file replacements and nothing else using this exact format:

            ### path/to/file
            ```language-identifier
            complete replacement contents
            ```

            The `###` heading must be immediately followed by its code fence. Each path may appear only once. Omit files that should remain unchanged.

            Allowed path and language pairs:
            {allowed_files}
        """).format(allowed_files=allowed_files).strip()

    def _format_diagnostic_prompt(
        self,
        context: ParentRunContext,
        feedback_prompt: str | None = None,
    ) -> str:
        candidate_files = ", ".join(
            f"`{spec.file}`" for spec in self.evo_config.evolvable_files
        )
        prompt = textwrap.dedent("""
            The last code update did not beat the parent score of **{score}**.
            Write instrumented diagnostic replacements for one or more candidate files to learn why.

            Candidate files: {candidate_files}

            Requirements:
            1. Preserve the input, output, and entrypoint behavior required by `{evaluate_file}`.
            2. Add lightweight instrumentation that prints targeted, actionable evidence.
            3. Go beyond trivial shape or head printing when the program uses structured data.
            4. Prefer observations that can motivate a concrete next code change.
            5. Do not try to improve the score in this step; this step is for diagnosis only.
            6. Keep the diagnostic run brief and low cost.
        """).format(
            score=context.parent_score,
            candidate_files=candidate_files,
            evaluate_file=self.evo_config.evaluate_file,
        ).strip()
        if feedback_prompt:
            prompt = feedback_prompt.rstrip() + "\n\n" + prompt
        return prompt + "\n\n" + self._format_file_replacement_contract()

    def _format_success_summary_prompt(self, session: ModelSession) -> str:
        return textwrap.dedent("""
            A new program has been accepted.

            Final evaluation feedback:
            {feedback_prompt}

            Based on the changes that successfully improved the combined_score, provide a general, reusable high-level strategy
            that made those changes effective. The strategy should not be specific to this program and should be applicable to future iterations.

            End the response with exactly:
            Strategy: <concise high-level strategy>
        """).format(
            feedback_prompt=session.success_feedback_prompt,
        ).strip()

    def _format_failure_summary_prompt(self, session: ModelSession) -> str:
        return textwrap.dedent("""
            You were unable to improve the program.

            Final evaluation feedback:
            {feedback_prompt}

            Concisely explain why.
        """).format(
            feedback_prompt=session.last_evaluation_feedback or "No program evaluation feedback is available.",
        ).strip()

    def _extract_turn_payload(self, response_text: str) -> dict[str, str]:
        return extract_file_replacements(
            response_text,
            {
                spec.file: spec.lang_identifier
                for spec in self.evo_config.evolvable_files
            },
        )

    def _format_missing_fence_prompt(
        self,
        context: ParentRunContext,
        session: ModelSession,
        exc: ValueError,
    ) -> str:
        response_kind = session.scheduled_turn.expected_response
        response_label = {
            ModelResponseKind.CODE_UPDATE: "code update",
            ModelResponseKind.DIAGNOSE: "diagnostic code",
        }[response_kind]
        prompt = textwrap.dedent("""
            Your previous {response_label} response could not be parsed.

            Parser error: {error}

            {replacement_contract}
        """).format(
            response_label=response_label,
            error=exc,
            replacement_contract=self._format_file_replacement_contract(),
        ).strip()
        if response_kind not in {
            ModelResponseKind.CODE_UPDATE,
            ModelResponseKind.DIAGNOSE,
        }:
            raise RuntimeError(f"Unexpected response kind: {response_kind}")
        return prompt

    def _append_captured_output(
        self,
        lines: list[str],
        results: dict[str, Any],
        source_name: str,
    ) -> None:
        stdout = results.get("stdout_log", "").strip()
        stderr = results.get("stderr_log", "").strip()
        if stdout != "":
            lines.append(f"Here is the standard output of the {source_name}:")
            lines.append("```")
            lines.append(stdout)
            lines.append("```")
        if stderr != "":
            lines.append(f"Here is the standard error of the {source_name}:")
            lines.append("```")
            lines.append(stderr)
            lines.append("```")

    def _raise_if_result_preempted_or_stale(
        self,
        context: ParentRunContext,
        result: dict[str, Any],
        activity: str,
    ) -> None:
        # Backend preemption and DB staleness mean this claim should be abandoned
        # immediately; normal helper return values would be ambiguous.
        if result.get("preempted"):
            logger.info(
                f"Worker {self.worker_id}: {activity} preempted because the claim became stale."
            )
            raise StaleClaimError
        self._raise_if_claim_stale(
            context,
            f"{activity} result is stale for claim {{claim_id}}.",
        )

    def _raise_if_claim_stale(
        self,
        context: ParentRunContext,
        message_template: str,
    ) -> None:
        claim_is_stale: bool = ray.get(
            self.db.should_preempt_claim.remote(context.claim_id)
        )
        if not claim_is_stale:
            return

        logger.info(
            f"Worker {self.worker_id}: "
            + message_template.format(
                claim_id=context.claim_id,
            )
        )
        raise StaleClaimError
