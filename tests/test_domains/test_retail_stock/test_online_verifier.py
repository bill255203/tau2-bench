"""Minimal retail-stock validation for transactional agent verification."""

import json
from typing import Optional

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import TextRunConfig
from tau2.domains.retail_stock.environment import get_environment, get_tasks
from tau2.environment.tool import as_tool
from tau2.orchestrator.orchestrator import (
    Orchestrator,
    Role,
    build_trusted_event_log,
)
from tau2.runner.build import build_text_orchestrator
from tau2.user.user_simulator import DummyUser
from tau2.verifier.base import Verifier
from tau2.verifier.llm_verifier import SYSTEM_PROMPT, LLMVerifier
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
        history: list[Message],
        proposed_message: AssistantMessage,
        trusted_event_log=None,
        tools=None,
    ) -> VerifierVerdict:
        self.calls.append(
            {
                "policy": policy,
                "history": history,
                "trusted_event_log": trusted_event_log,
                "proposal": proposed_message,
                "tools": tools,
            }
        )
        return VerifierVerdict(
            decision=self.decision,
            reason=f"test verdict: {self.decision.value}",
            confidence=1.0,
        )


class SequencedVerifier(RecordingVerifier):
    """Return ordered verdicts so a revise path can be tested deterministically."""

    def __init__(self, verdicts: list[VerifierVerdict]):
        super().__init__(verdicts[0].decision)
        self.verdicts = verdicts

    def verify(self, **kwargs) -> VerifierVerdict:
        self.calls.append(
            {
                "policy": kwargs["policy"],
                "history": kwargs["history"],
                "trusted_event_log": kwargs.get("trusted_event_log"),
                "proposal": kwargs["proposed_message"],
                "tools": kwargs.get("tools"),
            }
        )
        return self.verdicts[len(self.calls) - 1]


class RevisingStockLookupAgent(StockLookupAgent):
    """Propose an invalid call first, then use private verifier feedback."""

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: dict
    ) -> tuple[AssistantMessage, dict]:
        state["proposals"] += 1
        return (
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="rejected-call",
                        name="not_a_real_tool",
                        arguments={},
                    )
                ],
            ),
            state,
        )

    def generate_next_message_with_verifier_feedback(
        self,
        message: ValidAgentInputMessage,
        state: dict,
        feedback: str,
    ) -> tuple[AssistantMessage, dict]:
        state["proposals"] += 1
        state["verifier_feedback"] = feedback
        return (
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="revised-stock-call",
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
    assert verifier.calls[0]["policy"] == orchestrator.environment.get_policy()
    assert verifier.calls[0]["history"] == orchestrator.trajectory[:-1]
    assert verifier.calls[0]["trusted_event_log"] == []
    assert verifier.calls[0]["proposal"].tool_calls[0].name == "get_variant_stock"
    assert [tool.name for tool in verifier.calls[0]["tools"]] == ["get_variant_stock"]
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


def test_revision_feedback_replaces_uncommitted_proposal() -> None:
    verifier = SequencedVerifier(
        [
            VerifierVerdict(
                decision=VerifierDecision.REVISE,
                reason="The tool does not exist.",
                revision_instruction="Use get_variant_stock with the requested IDs.",
            ),
            VerifierVerdict(
                decision=VerifierDecision.PASS,
                reason="The replacement is grounded and valid.",
            ),
        ]
    )
    orchestrator = _orchestrator(verifier)
    agent = RevisingStockLookupAgent(
        tools=orchestrator.environment.get_tools(),
        domain_policy=orchestrator.environment.get_policy(),
    )
    orchestrator.agent = agent
    orchestrator.agent_state = agent.get_init_state()

    orchestrator.step()

    assert len(verifier.calls) == 2
    assert verifier.calls[0]["proposal"].tool_calls[0].name == "not_a_real_tool"
    assert verifier.calls[1]["proposal"].tool_calls[0].name == "get_variant_stock"
    assert len(orchestrator.verifier_records) == 2
    assert [record.revision_attempt for record in orchestrator.verifier_records] == [
        0,
        1,
    ]
    assert orchestrator.agent_state == {
        "proposals": 1,
        "verifier_feedback": "Use get_variant_stock with the requested IDs.",
    }
    assert orchestrator.trajectory[-1].tool_calls[0].name == "get_variant_stock"
    assert not orchestrator.environment.tools.db.event_log

    # Only the accepted replacement can reach the environment.
    orchestrator.step()
    assert len(orchestrator.environment.tools.db.event_log) == 1


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
    verifier = LLMVerifier("test/verifier-model", {"temperature": 0.0})
    history = [
        UserMessage(
            role="user",
            content="Check this variant's stock.",
            usage={"prompt_tokens": 999},
            raw_data={"reasoning_content": "private reasoning must not be sent"},
        )
    ]
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
        history=history,
        proposed_message=proposal,
    )

    assert verdict.decision == VerifierDecision.PASS
    assert verdict.confidence == 0.98
    assert captured["model"] == "test/verifier-model"
    assert captured["call_name"] == "agent_output_verifier"
    assert captured["temperature"] == 0.0
    verifier_payload = captured["messages"][1].content
    assert "private reasoning must not be sent" not in verifier_payload
    assert "raw_data" not in verifier_payload
    assert "prompt_tokens" not in verifier_payload
    assert "Check this variant's stock." in verifier_payload
    assert "get_variant_stock" in verifier_payload


def test_verifier_prompt_scopes_protocol_rules_to_current_proposal() -> None:
    assert "per-message protocol rules only to proposed_agent_output" in SYSTEM_PROMPT
    assert "earlier committed calls is allowed" in SYSTEM_PROMPT
    assert "both non-empty text" in SYSTEM_PROMPT
    assert "does not complete every remaining user goal" in SYSTEM_PROMPT


def test_llm_verifier_separates_conversation_from_trusted_events(
    monkeypatch,
) -> None:
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return AssistantMessage(
            role="assistant",
            content='{"decision":"pass","reason":"event is authoritative"}',
        )

    monkeypatch.setattr("tau2.verifier.llm_verifier.generate", fake_generate)
    verifier = LLMVerifier("test/verifier-model")
    history = [
        UserMessage(role="user", content="Check order #W123."),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="order-call",
                    name="get_order_details",
                    arguments={"order_id": "#W123"},
                )
            ],
        ),
        ToolMessage(
            id="order-call",
            role="tool",
            content='{"status":"delivered"}',
            error=False,
        ),
    ]

    verifier.verify(
        policy="Delivered orders cannot be modified.",
        history=history,
        trusted_event_log=build_trusted_event_log(history),
        proposed_message=AssistantMessage(role="assistant", content="It is delivered."),
    )

    payload = json.loads(captured["messages"][1].content)
    assert payload["relevant_conversation"] == [
        {"role": "user", "content": "Check order #W123."}
    ]
    assert payload["trusted_event_log"] == [
        {
            "sequence": 0,
            "call_id": "order-call",
            "tool_name": "get_order_details",
            "arguments": {"order_id": "#W123"},
            "requestor": "assistant",
            "result": '{"status":"delivered"}',
            "success": True,
            "executed_at": history[2].timestamp,
        }
    ]


def test_llm_verifier_input_excludes_hidden_task_fields(monkeypatch) -> None:
    captured_messages = []

    def fake_generate(**kwargs):
        captured_messages.extend(kwargs["messages"])
        return AssistantMessage(
            role="assistant",
            content=(
                '{"decision":"pass","reason":"grounded stock lookup",'
                '"revision_instruction":null,"confidence":1.0}'
            ),
        )

    monkeypatch.setattr("tau2.verifier.llm_verifier.generate", fake_generate)
    orchestrator = _orchestrator(LLMVerifier("test/verifier-model", min_tool_calls=0))
    task = orchestrator.task
    # The task must carry hidden fields for this test to mean anything.
    assert task.evaluation_criteria is not None
    assert task.user_scenario is not None

    orchestrator.step()

    assert len(orchestrator.verifier_records) == 1
    assert set(json.loads(captured_messages[1].content)) == {
        "policy",
        "relevant_conversation",
        "trusted_event_log",
        "proposed_agent_output",
    }
    verifier_input = "\n".join(message.content for message in captured_messages)
    assert "evaluation_criteria" not in verifier_input
    assert "user_scenario" not in verifier_input
    # Values from the hidden fields must not leak under other keys either.
    assert task.id not in verifier_input
    assert task.user_scenario.instructions.task_instructions not in verifier_input


def test_llm_verifier_uses_read_tool_before_revising(monkeypatch) -> None:
    """A verifier can inspect authoritative state before rejecting a proposal."""
    calls = []

    def inspect_order(order_id: str) -> dict:
        """Inspect an order without modifying it.

        Args:
            order_id: The order to inspect.

        Returns:
            Current order state.
        """
        assert order_id == "#W123"
        return {"order_id": order_id, "status": "delivered"}

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="verify-order",
                        name="inspect_order",
                        arguments={"order_id": "#W123"},
                    )
                ],
            )
        return AssistantMessage(
            role="assistant",
            content=json.dumps(
                {
                    "decision": "revise",
                    "reason": "The order is delivered, not pending.",
                    "revision_instruction": "Do not modify this delivered order.",
                    "confidence": 0.99,
                    "checks_performed": [
                        {
                            "claim": "The order is modifiable",
                            "result": "contradicted",
                            "evidence": "inspect_order returned status=delivered",
                        }
                    ],
                    "violated_rule": "Only pending orders may be modified.",
                    "unmet_goals": [],
                }
            ),
        )

    monkeypatch.setattr("tau2.verifier.llm_verifier.generate", fake_generate)
    verifier = LLMVerifier("test/verifier-model", max_tool_calls=2, min_tool_calls=1)
    proposal = AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="modify-order",
                name="modify_pending_order_address",
                arguments={"order_id": "#W123"},
            )
        ],
    )

    verdict = verifier.verify(
        policy="Only pending orders may be modified.",
        history=[UserMessage(role="user", content="Change my order address.")],
        proposed_message=proposal,
        tools=[as_tool(inspect_order)],
    )

    assert verdict.decision == VerifierDecision.REVISE
    assert verdict.checks_performed[0].result == "contradicted"
    assert len(calls) == 2
    assert calls[0]["tools"][0].name == "inspect_order"
    assert calls[0]["tool_choice"] == "required"
    assert calls[1]["tool_choice"] == "auto"
    tool_result = calls[1]["messages"][-1]
    assert tool_result.role == "tool"
    assert '"status": "delivered"' in tool_result.content


def test_llm_verifier_reprompts_when_required_tool_is_skipped(monkeypatch) -> None:
    """A provider that ignores required tool choice gets one corrective retry."""
    calls = []

    def inspect_order(order_id: str) -> dict:
        """Inspect an order.

        Args:
            order_id: The order to inspect.

        Returns:
            Current order state.
        """
        return {"order_id": order_id, "status": "delivered"}

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return AssistantMessage(
                role="assistant",
                content=(
                    '{"decision":"revise","reason":"status unknown",'
                    '"revision_instruction":"inspect it"}'
                ),
            )
        if len(calls) == 2:
            return AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="verify-order",
                        name="inspect_order",
                        arguments={"order_id": "#W123"},
                    )
                ],
            )
        return AssistantMessage(
            role="assistant",
            content=(
                '{"decision":"revise","reason":"order is delivered",'
                '"revision_instruction":"do not modify it"}'
            ),
        )

    monkeypatch.setattr("tau2.verifier.llm_verifier.generate", fake_generate)
    verifier = LLMVerifier("test/verifier-model", max_tool_calls=2, min_tool_calls=1)
    verdict = verifier.verify(
        policy="Only pending orders may be modified.",
        history=[UserMessage(role="user", content="Change order #W123.")],
        proposed_message=AssistantMessage(
            role="assistant",
            content="I will change the address.",
        ),
        tools=[as_tool(inspect_order)],
    )

    assert verdict.decision == VerifierDecision.REVISE
    assert len(calls) == 3
    assert any(
        "required read-only check" in (message.content or "")
        for message in calls[1]["messages"]
        if isinstance(message, UserMessage)
    )
