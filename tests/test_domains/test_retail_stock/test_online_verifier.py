"""Minimal retail-stock validation for transactional agent verification."""

from typing import Optional

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import AssistantMessage, Message, ToolCall, UserMessage
from tau2.data_model.simulation import TextRunConfig
from tau2.domains.retail_stock.environment import get_environment, get_tasks
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.runner.build import build_text_orchestrator
from tau2.user.user_simulator import DummyUser
from tau2.verifier.base import Verifier
from tau2.verifier.llm_verifier import LLMVerifier
from tau2.verifier.models import VerifierDecision, VerifierVerdict


class StockLookupAgent(HalfDuplexAgent[dict]):
    """Deterministic agent that proposes the retail-stock lookup tool."""

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> dict:
        return {"proposals": 0}

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: dict
    ) -> tuple[AssistantMessage, dict]:
        state["proposals"] += 1
        return (
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="stock-call",
                        name="get_variant_stock",
                        arguments={
                            "product_id": "9523456873",
                            "item_id": "9612497925",
                        },
                    )
                ],
            ),
            state,
        )


class RecordingVerifier(Verifier):
    """Record verifier inputs and return a configured decision."""

    def __init__(self, decision: VerifierDecision):
        self.decision = decision
        self.calls: list[dict] = []

    def verify(
        self,
        *,
        policy: str,
        task,
        history: list[Message],
        proposed_message: AssistantMessage,
    ) -> VerifierVerdict:
        self.calls.append(
            {
                "policy": policy,
                "task_id": task.id,
                "history": history,
                "proposal": proposed_message,
            }
        )
        return VerifierVerdict(
            decision=self.decision,
            reason=f"test verdict: {self.decision.value}",
            confidence=1.0,
        )


def _orchestrator(verifier: Verifier) -> Orchestrator:
    environment = get_environment()
    agent = StockLookupAgent(
        tools=environment.get_tools(), domain_policy=environment.get_policy()
    )
    orchestrator = Orchestrator(
        domain="retail_stock",
        agent=agent,
        user=DummyUser(),
        environment=environment,
        task=get_tasks()[0],
        verifier=verifier,
    )
    orchestrator.initialize()
    user_message = UserMessage(
        role="user",
        content="Do you have 10 units of the requested shirt variant?",
    )
    orchestrator.trajectory.append(user_message)
    orchestrator.message = user_message
    orchestrator.from_role = Role.USER
    orchestrator.to_role = Role.AGENT
    return orchestrator


def test_passing_verifier_commits_retail_stock_proposal() -> None:
    verifier = RecordingVerifier(VerifierDecision.PASS)
    orchestrator = _orchestrator(verifier)

    orchestrator.step()

    assert len(verifier.calls) == 1
    assert verifier.calls[0]["task_id"] == orchestrator.task.id
    assert verifier.calls[0]["history"] == orchestrator.trajectory[:-1]
    assert verifier.calls[0]["proposal"].tool_calls[0].name == "get_variant_stock"
    assert orchestrator.agent_state == {"proposals": 1}
    assert orchestrator.trajectory[-1].tool_calls[0].name == "get_variant_stock"
    assert orchestrator.to_role == Role.ENV
    assert len(orchestrator.verifier_records) == 1

    # The committed proposal routes normally and executes on the next step.
    orchestrator.step()
    assert len(orchestrator.environment.tools.db.event_log) == 1


def test_rejected_verifier_discards_retail_stock_proposal() -> None:
    verifier = RecordingVerifier(VerifierDecision.BLOCK)
    orchestrator = _orchestrator(verifier)
    trajectory_before = list(orchestrator.trajectory)

    orchestrator.step()

    assert len(verifier.calls) == 1
    assert orchestrator.agent_state == {"proposals": 0}
    assert orchestrator.trajectory == trajectory_before
    assert not orchestrator.environment.tools.db.event_log
    assert orchestrator.done
    assert orchestrator.termination_reason.value == "verifier_rejected"


def test_retail_stock_builder_enables_llm_verifier() -> None:
    config = TextRunConfig(
        domain="retail_stock",
        verifier_model="test/verifier-model",
        verifier_llm_args={"temperature": 0.0},
    )

    orchestrator = build_text_orchestrator(config, get_tasks()[0])

    assert isinstance(orchestrator.verifier, LLMVerifier)
    assert orchestrator.verifier.llm == "test/verifier-model"
    assert orchestrator.verifier.llm_args == {"temperature": 0.0}


def test_llm_verifier_parses_structured_result(monkeypatch) -> None:
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return AssistantMessage(
            role="assistant",
            content=(
                '{"decision":"pass","reason":"grounded stock lookup",'
                '"revision_instruction":null,"confidence":0.98}'
            ),
        )

    monkeypatch.setattr("tau2.verifier.llm_verifier.generate", fake_generate)
    task = get_tasks()[0]
    verifier = LLMVerifier("test/verifier-model", {"temperature": 0.0})
    history = [UserMessage(role="user", content="Check this variant's stock.")]
    proposal = AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="stock-call",
                name="get_variant_stock",
                arguments={
                    "product_id": "9523456873",
                    "item_id": "9612497925",
                },
            )
        ],
    )

    verdict = verifier.verify(
        policy=get_environment().get_policy(),
        task=task,
        history=history,
        proposed_message=proposal,
    )

    assert verdict.decision == VerifierDecision.PASS
    assert verdict.confidence == 0.98
    assert captured["model"] == "test/verifier-model"
    assert captured["call_name"] == "agent_output_verifier"
    assert captured["temperature"] == 0.0
