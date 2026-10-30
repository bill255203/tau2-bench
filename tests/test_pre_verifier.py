"""Tests for the pre-verifier (rules v1 and v2) and its wiring."""

import inspect
import json
from copy import deepcopy
from typing import Optional

import pytest
from pydantic import ValidationError

from tau2 import cli
from tau2.agent.llm_agent import LLMAgent
from tau2.agent.pre_verifier import prepare, select_order_tools, tool_chars
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import TextRunConfig
from tau2.domains.airline.environment import get_environment as get_airline_env
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.environment.tool import as_tool
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.runner import build as build_module
from tau2.runner.build import build_agent, build_text_orchestrator
from tau2.runner.helpers import get_info
from tau2.user.user_simulator import DummyUser
from tau2.verifier.base import Verifier
from tau2.verifier.models import VerifierDecision, VerifierVerdict

IDENTITY_STAGE = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "transfer_to_human_agents",
}
ALWAYS_AFTER_IDENTITY = {
    "get_user_details",
    "get_order_details",
    "get_product_details",
    "get_item_details",
    "list_all_product_types",
    "calculate",
    "modify_user_address",
    "transfer_to_human_agents",
}
PENDING_WRITES = {
    "cancel_pending_order",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
}
DELIVERED_WRITES = {"return_delivered_order_items", "exchange_delivered_order_items"}


@pytest.fixture(scope="module")
def environment():
    return get_environment()


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _call(name: str, call_id: str, **arguments) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


def _result(call_id: str, content, error: bool = False) -> ToolMessage:
    if not isinstance(content, str):
        content = json.dumps(content)
    return ToolMessage(
        id=call_id, role="tool", requestor="assistant", content=content, error=error
    )


def _order(order_id: str, status: str) -> dict:
    return {"order_id": order_id, "user_id": "user_1", "status": status, "items": []}


def _identified() -> list[Message]:
    return [
        UserMessage(role="user", content="Hi, I need help with an order."),
        _call("find_user_id_by_email", "c1", email="a@example.com"),
        _result("c1", "user_1"),
    ]


def _order_lookup(order_id: str, status: str, call_id: str) -> list[Message]:
    return [
        _call("get_order_details", call_id, order_id=order_id),
        _result(call_id, _order(order_id, status)),
    ]


def _offered(environment, messages) -> tuple[set[str], object]:
    _, tools, record = prepare(messages, environment.get_tools(), "policy")
    return {tool.name for tool in tools}, record


# ---------------------------------------------------------------------------
# prepare(): rule v1
# ---------------------------------------------------------------------------


def test_before_identity_offers_lookup_and_transfer_only(environment):
    offered, record = _offered(
        environment, [UserMessage(role="user", content="Cancel my order please.")]
    )

    assert offered == IDENTITY_STAGE
    assert record.applied
    assert not record.identity_verified
    assert (
        set(record.removed_tools)
        == {tool.name for tool in environment.get_tools()} - IDENTITY_STAGE
    )
    assert record.tool_chars_after < record.tool_chars_before


def test_failed_lookup_does_not_confirm_identity(environment):
    messages = [
        UserMessage(role="user", content="My email is a@example.com"),
        _call("find_user_id_by_email", "c1", email="a@example.com"),
        _result("c1", "Error: User not found", error=True),
    ]

    offered, record = _offered(environment, messages)

    assert offered == IDENTITY_STAGE
    assert not record.identity_verified


def test_after_identity_without_orders_hides_order_writes(environment):
    offered, record = _offered(environment, _identified())

    assert offered == ALWAYS_AFTER_IDENTITY
    assert record.identity_verified
    assert record.order_statuses == {}


@pytest.mark.parametrize(
    ("status", "expected_writes"),
    [
        ("pending", PENDING_WRITES),
        (
            "pending (item modified)",
            {"modify_pending_order_address", "modify_pending_order_payment"},
        ),
        ("delivered", DELIVERED_WRITES),
        ("processed", set()),
        ("cancelled", set()),
        ("return requested", set()),
        ("exchange requested", set()),
    ],
)
def test_order_writes_follow_environment_status_checks(
    environment, status, expected_writes
):
    offered, record = _offered(
        environment, _identified() + _order_lookup("#W1", status, "c2")
    )

    assert offered == ALWAYS_AFTER_IDENTITY | expected_writes
    assert record.order_statuses == {"#W1": status}


def test_orders_combine_and_latest_result_wins(environment):
    messages = (
        _identified()
        + _order_lookup("#W1", "pending", "c2")
        + _order_lookup("#W2", "processed", "c3")
        + _order_lookup("#W1", "delivered", "c4")
    )

    offered, record = _offered(environment, messages)

    assert offered == ALWAYS_AFTER_IDENTITY | DELIVERED_WRITES
    assert record.order_statuses == {"#W1": "delivered", "#W2": "processed"}


def test_successful_write_updates_status_and_failed_write_does_not(environment):
    cancelled = (
        _identified()
        + _order_lookup("#W1", "pending", "c2")
        + [
            _call(
                "cancel_pending_order", "c3", order_id="#W1", reason="no longer needed"
            ),
            _result("c3", _order("#W1", "cancelled")),
        ]
    )
    failed = (
        _identified()
        + _order_lookup("#W1", "pending", "c2")
        + [
            _call("cancel_pending_order", "c3", order_id="#W1", reason="other"),
            _result("c3", "Error: Invalid reason", error=True),
        ]
    )

    offered_cancelled, _ = _offered(environment, cancelled)
    offered_failed, _ = _offered(environment, failed)

    assert offered_cancelled == ALWAYS_AFTER_IDENTITY
    assert offered_failed == ALWAYS_AFTER_IDENTITY | PENDING_WRITES


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        _order("#W1", "on hold"),
        {"unexpected": "shape"},
        {"order_id": "#W1", "status": []},
        {"order_id": "#W1", "status": {}},
        {"order_id": "#W1", "status": None},
        {"order_id": "#W1", "status": 3},
    ],
)
def test_unreadable_order_result_offers_every_order_write(environment, content):
    messages = _identified() + [
        _call("get_order_details", "c2", order_id="#W1"),
        _result("c2", content),
    ]

    offered, record = _offered(environment, messages)

    assert offered == ALWAYS_AFTER_IDENTITY | PENDING_WRITES | DELIVERED_WRITES
    assert record.unreadable_order_results == 1
    assert record.unknown_status_orders == ["#W1"]


def test_unreadable_result_without_order_id_offers_every_order_write(environment):
    messages = _identified() + [
        _call("get_order_details", "c2"),
        _result("c2", "not json"),
    ]

    offered, record = _offered(environment, messages)

    assert offered == ALWAYS_AFTER_IDENTITY | PENDING_WRITES | DELIVERED_WRITES
    assert record.unattributed_order_results == 1


@pytest.mark.parametrize(
    "content",
    [
        {"status": "cancelled"},
        {"order_id": None, "status": "cancelled"},
        {"order_id": [], "status": "cancelled"},
    ],
)
def test_result_without_order_id_uses_the_call_argument(environment, content):
    messages = (
        _identified()
        + _order_lookup("#W1", "pending", "c2")
        + [_call("get_order_details", "c3", order_id="#W1"), _result("c3", content)]
    )

    offered, record = _offered(environment, messages)

    assert offered == ALWAYS_AFTER_IDENTITY
    assert record.order_statuses == {"#W1": "cancelled"}
    assert record.unattributed_order_results == 0


def test_readable_status_with_no_order_id_anywhere_offers_every_order_write(
    environment,
):
    messages = _identified() + [
        _call("get_order_details", "c2"),
        _result("c2", {"status": "cancelled"}),
    ]

    offered, record = _offered(environment, messages)

    assert offered == ALWAYS_AFTER_IDENTITY | PENDING_WRITES | DELIVERED_WRITES
    assert record.unattributed_order_results == 1
    assert record.unreadable_order_results == 0


def test_newer_readable_status_replaces_an_unreadable_one(environment):
    messages = (
        _identified()
        + _order_lookup("#W1", "on hold", "c2")
        + _order_lookup("#W1", "cancelled", "c3")
    )

    offered, record = _offered(environment, messages)

    assert offered == ALWAYS_AFTER_IDENTITY
    assert record.order_statuses == {"#W1": "cancelled"}
    assert record.unknown_status_orders == []
    assert record.unreadable_order_results == 1


def test_results_pair_by_position_when_ids_are_missing(environment):
    messages = [
        UserMessage(role="user", content="Order #W1, email a@example.com"),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="", name="find_user_id_by_email", arguments={}),
                ToolCall(id="", name="get_order_details", arguments={}),
            ],
        ),
        _result("", "user_1"),
        _result("", _order("#W1", "delivered")),
    ]

    offered, record = _offered(environment, messages)

    assert record.identity_verified
    assert offered == ALWAYS_AFTER_IDENTITY | DELIVERED_WRITES


def test_results_pair_by_id_when_ids_are_mixed(environment):
    messages = [
        UserMessage(role="user", content="Order #W1, email a@example.com"),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="i", name="find_user_id_by_email", arguments={}),
                ToolCall(id="", name="get_order_details", arguments={}),
            ],
        ),
        # The order result arrives first; it must not answer the identity call.
        _result("", _order("#W1", "pending")),
        _result("i", "Error: User not found", error=True),
    ]

    offered, record = _offered(environment, messages)

    assert not record.identity_verified
    assert offered == IDENTITY_STAGE
    assert record.order_statuses == {"#W1": "pending"}


def test_user_requested_calls_are_not_paired_with_agent_results(environment):
    messages = [
        UserMessage(role="user", content="Order #W1, email a@example.com"),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="", name="find_user_id_by_email", arguments={}, requestor="user"
                ),
                ToolCall(id="", name="get_order_details", arguments={}),
            ],
        ),
        ToolMessage(
            id="", role="tool", requestor="user", content="Error: no user", error=True
        ),
        _result("", _order("#W1", "pending")),
    ]

    _, record = _offered(environment, messages)

    assert not record.identity_verified


def test_unpaired_result_leaves_the_request_unchanged(environment):
    messages = _identified() + [_result("no-such-call", _order("#W1", "pending"))]
    tools = environment.get_tools()

    _, out_tools, record = prepare(messages, tools, "policy")

    assert [tool.name for tool in out_tools] == [tool.name for tool in tools]
    assert not record.applied
    assert record.reason == "unpaired_tool_result"


def test_prepare_does_not_modify_its_inputs(environment):
    messages = _identified() + _order_lookup("#W1", "pending", "c2")
    tools = environment.get_tools()
    messages_before = deepcopy(messages)
    tool_names_before = [tool.name for tool in tools]

    out_messages, out_tools, record = prepare(messages, tools, "policy")

    assert messages == messages_before
    assert [tool.name for tool in tools] == tool_names_before
    assert out_messages == messages and out_messages is not messages
    # Offered tools keep the original order.
    assert [tool.name for tool in out_tools] == [
        name for name in tool_names_before if name in record.offered_tools
    ]
    assert record.message_chars_after == record.message_chars_before
    assert record.tool_chars_after == tool_chars(out_tools)


def test_unsupported_toolset_is_left_unchanged():
    airline = get_airline_env()
    tools = airline.get_tools()

    _, out_tools, record = prepare(
        [UserMessage(role="user", content="Change my flight")], tools, "policy"
    )

    assert [tool.name for tool in out_tools] == [tool.name for tool in tools]
    assert not record.applied
    assert record.reason == "unsupported_toolset"
    assert record.removed_tools == []


def test_unknown_tools_are_always_offered(environment):
    def get_variant_stock(item_id: str) -> str:
        """Return the stock of an item variant."""
        return "10"

    tools = environment.get_tools() + [as_tool(get_variant_stock)]

    _, out_tools, _ = prepare(
        [UserMessage(role="user", content="Is it in stock?")], tools, "policy"
    )

    assert {tool.name for tool in out_tools} == IDENTITY_STAGE | {"get_variant_stock"}


def test_prepare_inputs_exclude_task_data():
    assert list(inspect.signature(prepare).parameters) == [
        "messages",
        "tools",
        "policy",
        "rule",
    ]


def test_unknown_rule_is_rejected(environment):
    with pytest.raises(ValueError):
        prepare([], environment.get_tools(), "policy", rule="v9")
    with pytest.raises(ValueError):
        LLMAgent(
            tools=environment.get_tools(),
            domain_policy="policy",
            llm="test/agent",
            pre_verifier="v9",
        )
    with pytest.raises(ValidationError):
        TextRunConfig(domain="retail", pre_verifier="v9")


# ---------------------------------------------------------------------------
# LLMAgent wiring
# ---------------------------------------------------------------------------


class FakeGenerate:
    """Scripted replacement for ``generate``; an Exception reply is raised."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def __call__(self, **kwargs) -> AssistantMessage:
        self.calls.append({**kwargs, "messages": deepcopy(kwargs["messages"])})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _text(content: Optional[str], cost: Optional[float] = None) -> AssistantMessage:
    return AssistantMessage(role="assistant", content=content, cost=cost)


def _agent(
    environment,
    monkeypatch,
    replies,
    pre_verifier: Optional[str] = "v1",
    **agent_kwargs,
):
    fake = FakeGenerate(replies)
    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake)
    agent = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm="test/agent",
        pre_verifier=pre_verifier,
        **agent_kwargs,
    )
    return agent, fake


def test_agent_off_sends_the_full_request(environment, monkeypatch):
    agent, fake = _agent(environment, monkeypatch, [_text("Hello")], pre_verifier=None)
    state = agent.get_init_state()
    user_message = UserMessage(role="user", content="Hi")

    agent.generate_next_message(user_message, state)

    assert fake.calls[0]["tools"] is agent.tools
    assert fake.calls[0]["messages"] == state.system_messages + [user_message]
    assert agent.pop_pre_verifier_records() == []


def test_agent_on_trims_only_the_model_copy(environment, monkeypatch):
    agent, fake = _agent(environment, monkeypatch, [_text("Your email please?")])
    state = agent.get_init_state()
    user_message = UserMessage(role="user", content="Cancel my order")

    reply, state = agent.generate_next_message(user_message, state)

    assert {tool.name for tool in fake.calls[0]["tools"]} == IDENTITY_STAGE
    assert len(agent.tools) == len(environment.get_tools())
    assert fake.calls[0]["messages"] == state.system_messages + [user_message]
    assert state.messages == [user_message, reply]
    (record,) = agent.pop_pre_verifier_records()
    assert record.message_count == 2
    assert not record.revision
    assert agent.pop_pre_verifier_records() == []


def test_multi_tool_message_is_added_once(environment, monkeypatch):
    agent, fake = _agent(environment, monkeypatch, [_text("I see order #W1.")])
    history = [
        UserMessage(role="user", content="Email a@example.com, order #W1"),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="a", name="find_user_id_by_email", arguments={}),
                ToolCall(id="b", name="get_order_details", arguments={}),
            ],
        ),
    ]
    state = agent.get_init_state(message_history=history)
    results = MultiToolMessage(
        role="tool",
        tool_messages=[_result("a", "user_1"), _result("b", _order("#W1", "pending"))],
    )

    reply, state = agent.generate_next_message(results, state)

    assert state.messages == history + results.tool_messages + [reply]
    assert fake.calls[0]["messages"][1:] == history + results.tool_messages
    (record,) = agent.pop_pre_verifier_records()
    assert record.identity_verified
    assert record.order_statuses == {"#W1": "pending"}


def test_called_removed_tool_is_counted(environment, monkeypatch):
    hidden_call = _call("cancel_pending_order", "x", order_id="#W1", reason="other")
    agent, _ = _agent(environment, monkeypatch, [hidden_call])

    reply, _ = agent.generate_next_message(
        UserMessage(role="user", content="Cancel #W1"), agent.get_init_state()
    )

    assert reply.tool_calls[0].name == "cancel_pending_order"
    (record,) = agent.pop_pre_verifier_records()
    assert record.called_removed_tools == ["cancel_pending_order"]


def test_revision_keeps_feedback_and_marks_the_record(environment, monkeypatch):
    agent, fake = _agent(environment, monkeypatch, [_text("Revised"), _text("Next")])
    state = agent.get_init_state()

    agent.generate_next_message_with_verifier_feedback(
        UserMessage(role="user", content="Hi"), state, "Ask for the email first."
    )
    agent.generate_next_message(UserMessage(role="user", content="Ok"), state)

    sent = fake.calls[0]["messages"]
    assert isinstance(sent[1], SystemMessage)
    assert "Ask for the email first." in sent[1].content
    assert len(state.system_messages) == 1
    assert [record.revision for record in agent.pop_pre_verifier_records()] == [
        True,
        False,
    ]


# ---------------------------------------------------------------------------
# select_order_tools(): rule v2
# ---------------------------------------------------------------------------


def _two_orders() -> list[Message]:
    """Identified customer with a delivered #W1 and a pending #W2."""
    return (
        _identified()
        + _order_lookup("#W1", "delivered", "c2")
        + _order_lookup("#W2", "pending", "c3")
    )


def _select(environment, monkeypatch, messages, replies):
    fake = FakeGenerate(replies)
    monkeypatch.setattr("tau2.agent.pre_verifier.generate", fake)
    out_messages, tools, record = prepare(
        messages, environment.get_tools(), "policy", rule="v2"
    )
    tools = select_order_tools(out_messages, tools, record, llm="test/selector")
    return {tool.name for tool in tools}, record, fake


@pytest.mark.parametrize(
    "messages",
    [
        [UserMessage(role="user", content="Cancel my order please.")],
        _identified(),
        _identified() + _order_lookup("#W1", "delivered", "c2"),
        _two_orders(),
    ],
)
def test_v2_prepare_matches_v1(environment, messages):
    _, v1_tools, v1_record = prepare(messages, environment.get_tools(), "policy")
    _, v2_tools, v2_record = prepare(
        messages, environment.get_tools(), "policy", rule="v2"
    )

    assert [tool.name for tool in v2_tools] == [tool.name for tool in v1_tools]
    assert v2_record.rule == "v2"
    assert v2_record.model_dump(exclude={"rule"}) == v1_record.model_dump(
        exclude={"rule"}
    )


@pytest.mark.parametrize(
    "messages",
    [
        [UserMessage(role="user", content="Cancel my order please.")],
        _identified(),
        _identified() + _order_lookup("#W1", "processed", "c2"),
    ],
)
def test_v2_makes_no_model_call_with_fewer_than_two_candidates(
    environment, monkeypatch, messages
):
    offered, record, fake = _select(environment, monkeypatch, messages, [])
    v1_offered, _ = _offered(environment, messages)

    assert fake.calls == []
    assert offered == v1_offered
    assert record.model_candidates == []
    assert record.model_fallback is None


def test_v2_makes_no_model_call_for_unsupported_toolsets(monkeypatch):
    fake = FakeGenerate([])
    monkeypatch.setattr("tau2.agent.pre_verifier.generate", fake)
    tools = get_airline_env().get_tools()
    messages = [UserMessage(role="user", content="Change my flight")]
    _, out_tools, record = prepare(messages, tools, "policy", rule="v2")

    out_tools = select_order_tools(messages, out_tools, record, llm="test/selector")

    assert fake.calls == []
    assert out_tools == tools


@pytest.mark.parametrize(
    "reply",
    [
        '{"tools": ["return_delivered_order_items"]}',
        '```json\n{"tools": ["return_delivered_order_items"]}\n```',
    ],
)
def test_v2_model_drops_order_writes_that_do_not_fit(environment, monkeypatch, reply):
    messages = _two_orders() + [
        UserMessage(role="user", content="I want to return the lamp from #W1.")
    ]

    offered, record, fake = _select(
        environment, monkeypatch, messages, [_text(reply, cost=0.002)]
    )

    assert offered == ALWAYS_AFTER_IDENTITY | {"return_delivered_order_items"}
    assert set(record.model_candidates) == PENDING_WRITES | DELIVERED_WRITES
    assert record.model_kept == ["return_delivered_order_items"]
    assert record.model_raw_reply == reply
    assert record.model_cost == 0.002
    assert record.model_fallback is None
    assert set(record.offered_tools) == offered
    assert set(record.removed_tools) >= PENDING_WRITES | {
        "exchange_delivered_order_items"
    }
    assert not offered & set(record.removed_tools)
    assert record.tool_chars_after == tool_chars(
        [tool for tool in environment.get_tools() if tool.name in offered]
    )
    # The selector sees the conversation and the candidates only.
    (call,) = fake.calls
    assert call["model"] == "test/selector"
    system, request = call["messages"]
    assert isinstance(system, SystemMessage)
    payload = json.loads(request.content)
    assert [tool["name"] for tool in payload["candidate_tools"]] == (
        record.model_candidates
    )
    assert len(payload["conversation"]) == len(messages)
    assert payload["conversation"][-1]["content"] == messages[-1].content


@pytest.mark.parametrize(
    ("reply", "fallback"),
    [
        (_text("not json"), "unparsable"),
        (_text('{"answer": 1}'), "unparsable"),
        (_text("[1, 2]"), "unparsable"),
        (_text('{"tools": "cancel_pending_order"}'), "unparsable"),
        (_text('{"tools": []}'), "empty"),
        (_text('{"tools": ["get_user_details"]}'), "unknown_tool"),
        (_text('{"tools": ["cancel_pending_order", "refund_all"]}'), "unknown_tool"),
        (_text(None), "no_content"),
        (RuntimeError("rate limited"), "model_error: RuntimeError"),
    ],
)
def test_v2_unusable_reply_keeps_the_v1_set(environment, monkeypatch, reply, fallback):
    v1_offered, v1_record = _offered(environment, _two_orders())

    offered, record, _ = _select(environment, monkeypatch, _two_orders(), [reply])

    assert offered == v1_offered
    assert record.model_fallback == fallback
    assert record.model_kept == []
    assert record.removed_tools == v1_record.removed_tools
    assert record.tool_chars_after == v1_record.tool_chars_after


def test_v2_selector_inputs_exclude_task_data():
    assert list(inspect.signature(select_order_tools).parameters) == [
        "messages",
        "tools",
        "record",
        "llm",
        "llm_args",
    ]


@pytest.mark.parametrize(
    ("agent_kwargs", "selector_model", "selector_args"),
    [
        ({"llm_args": {"temperature": 0}}, "test/agent", {"temperature": 0}),
        (
            {
                "llm_args": {"temperature": 0},
                "pre_verifier_llm": "test/selector",
                "pre_verifier_llm_args": {"temperature": 0.5},
            },
            "test/selector",
            {"temperature": 0.5},
        ),
    ],
)
def test_agent_v2_sends_only_the_kept_tools(
    environment, monkeypatch, agent_kwargs, selector_model, selector_args
):
    hidden_call = _call("cancel_pending_order", "x", order_id="#W2", reason="other")
    agent, fake = _agent(
        environment, monkeypatch, [hidden_call], pre_verifier="v2", **agent_kwargs
    )
    selector = FakeGenerate([_text('{"tools": ["return_delivered_order_items"]}')])
    monkeypatch.setattr("tau2.agent.pre_verifier.generate", selector)
    state = agent.get_init_state(message_history=_two_orders())

    agent.generate_next_message(
        UserMessage(role="user", content="Return the lamp from #W1."), state
    )

    (call,) = selector.calls
    assert call["model"] == selector_model
    assert {key: call[key] for key in selector_args} == selector_args
    assert {tool.name for tool in fake.calls[0]["tools"]} == (
        ALWAYS_AFTER_IDENTITY | {"return_delivered_order_items"}
    )
    (record,) = agent.pop_pre_verifier_records()
    assert record.rule == "v2"
    assert record.model_kept == ["return_delivered_order_items"]
    assert record.called_removed_tools == ["cancel_pending_order"]


# ---------------------------------------------------------------------------
# Orchestrator records
# ---------------------------------------------------------------------------


class ScriptedVerifier(Verifier):
    def __init__(self, decisions: list[VerifierDecision]):
        self.decisions = list(decisions)

    def verify(self, **kwargs) -> VerifierVerdict:
        decision = self.decisions.pop(0)
        return VerifierVerdict(
            decision=decision,
            reason=f"test verdict: {decision.value}",
            revision_instruction="Ask for the email first."
            if decision == VerifierDecision.REVISE
            else None,
        )


def _orchestrator(environment, monkeypatch, replies, verifier=None):
    agent, fake = _agent(environment, monkeypatch, replies)
    orchestrator = Orchestrator(
        domain="retail",
        agent=agent,
        user=DummyUser(),
        environment=environment,
        task=get_tasks()[0],
        verifier=verifier,
    )
    orchestrator.initialize()
    user_message = UserMessage(role="user", content="I want to return an item.")
    orchestrator.trajectory.append(user_message)
    orchestrator.message = user_message
    orchestrator.from_role = Role.USER
    orchestrator.to_role = Role.AGENT
    return orchestrator, fake


def test_records_are_saved_without_a_post_verifier(environment, monkeypatch):
    orchestrator, _ = _orchestrator(environment, monkeypatch, [_text("Email?")])
    history_length = len(orchestrator.trajectory)

    orchestrator.step()

    info = orchestrator._build_verifier_info()
    assert set(info) == {"pre_verifier"}
    assert info["pre_verifier"]["rule"] == "v1"
    (record,) = info["pre_verifier"]["records"]
    assert record["history_length"] == history_length
    assert record["revision_attempt"] == 0


def test_called_removed_tool_still_runs(environment, monkeypatch):
    hidden_call = _call("list_all_product_types", "x")
    orchestrator, _ = _orchestrator(environment, monkeypatch, [hidden_call])

    orchestrator.step()  # agent proposes the removed tool
    orchestrator.step()  # environment executes it

    result = orchestrator.trajectory[-1]
    assert isinstance(result, ToolMessage) and not result.error
    assert isinstance(json.loads(result.content), dict)  # product types by name
    (record,) = orchestrator.pre_verifier_records
    assert record.called_removed_tools == ["list_all_product_types"]


def test_rejected_proposal_record_is_kept(environment, monkeypatch):
    orchestrator, _ = _orchestrator(
        environment,
        monkeypatch,
        [_text("Sure, done.")],
        verifier=ScriptedVerifier([VerifierDecision.BLOCK]),
    )
    state_before = deepcopy(orchestrator.agent_state)

    orchestrator.step()

    assert orchestrator.termination_reason.value == "verifier_rejected"
    assert orchestrator.agent_state == state_before
    info = orchestrator._build_verifier_info()
    (pre,) = info["pre_verifier"]["records"]
    (post,) = info["verifier"]["records"]
    assert pre["history_length"] == post["history_length"]
    assert pre["revision_attempt"] == post["revision_attempt"] == 0


def test_revision_records_both_attempts(environment, monkeypatch):
    orchestrator, fake = _orchestrator(
        environment,
        monkeypatch,
        [_text("Done."), _text("Could you share your email?")],
        verifier=ScriptedVerifier([VerifierDecision.REVISE, VerifierDecision.PASS]),
    )

    orchestrator.step()

    records = orchestrator.pre_verifier_records
    assert [record.revision_attempt for record in records] == [0, 1]
    assert [record.revision for record in records] == [False, True]
    assert orchestrator.trajectory[-1].content == "Could you share your email?"
    # The revision request carries the feedback but not the rejected proposal.
    contents = [message.content or "" for message in fake.calls[1]["messages"]]
    assert any("Ask for the email first." in content for content in contents)
    assert "Done." not in contents


class OneLineUser(DummyUser):
    """User that always says the same thing, so a full run can be driven."""

    def generate_next_message(self, message, state):
        return UserMessage(role="user", content="I want to return an item."), state


def test_simulation_run_info_survives_serialization(environment, monkeypatch):
    orchestrator, _ = _orchestrator(
        environment,
        monkeypatch,
        [_text("Sure, done.")],
        verifier=ScriptedVerifier([VerifierDecision.BLOCK]),
    )
    orchestrator.user = OneLineUser()

    run = orchestrator.run()

    info = json.loads(run.model_dump_json())["info"]
    assert info["pre_verifier"]["rule"] == "v1"
    assert len(info["pre_verifier"]["records"]) == 1
    assert len(info["verifier"]["records"]) == 1


def test_hidden_task_fields_never_reach_the_model_or_records(environment, monkeypatch):
    orchestrator, fake = _orchestrator(environment, monkeypatch, [_text("Email?")])
    task = orchestrator.task
    assert task.user_scenario is not None
    assert task.evaluation_criteria is not None

    orchestrator.step()

    sent = json.dumps([m.model_dump(mode="json") for m in fake.calls[0]["messages"]])
    saved = json.dumps(orchestrator._build_verifier_info())
    instructions = task.user_scenario.instructions.task_instructions
    for text in (sent, saved):
        assert "user_scenario" not in text
        assert "evaluation_criteria" not in text
        assert instructions not in text


# ---------------------------------------------------------------------------
# Config and builder
# ---------------------------------------------------------------------------


def test_cli_passes_the_rule_to_the_config(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "tau2.cli.run_domain", lambda config: captured.update(config=config)
    )
    monkeypatch.setattr(
        "sys.argv", ["tau2", "run", "--domain", "retail", "--pre-verifier", "v1"]
    )

    cli.main()

    assert captured["config"].pre_verifier == "v1"


def test_cli_passes_the_v2_model_to_the_config(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "tau2.cli.run_domain", lambda config: captured.update(config=config)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tau2",
            "run",
            "--domain",
            "retail",
            "--pre-verifier",
            "v2",
            "--pre-verifier-model",
            "test/selector",
            "--pre-verifier-llm-args",
            '{"temperature": 0}',
        ],
    )

    cli.main()

    config = captured["config"]
    assert config.pre_verifier == "v2"
    assert config.pre_verifier_model == "test/selector"
    assert config.pre_verifier_llm_args == {"temperature": 0}


def test_builder_passes_the_v2_model_to_the_agent():
    config = TextRunConfig(
        domain="retail",
        pre_verifier="v2",
        pre_verifier_model="test/selector",
        pre_verifier_llm_args={"temperature": 0},
    )

    agent = build_text_orchestrator(config, get_tasks()[0]).agent

    assert agent.pre_verifier == "v2"
    assert agent.pre_verifier_llm == "test/selector"
    assert agent.pre_verifier_llm_args == {"temperature": 0}
    assert get_info(config).agent_info.pre_verifier_model == "test/selector"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pre_verifier": None, "pre_verifier_llm": "test/selector"},
        {"pre_verifier": "v1", "pre_verifier_llm": "test/selector"},
        {"pre_verifier": "v2", "pre_verifier_llm_args": {"temperature": 0}},
    ],
)
def test_builder_rejects_a_pre_verifier_model_it_would_not_use(environment, kwargs):
    with pytest.raises(ValueError, match="(?i)pre-verifier model"):
        build_agent("llm_agent", environment, llm="test/agent", **kwargs)


def test_builder_passes_the_rule_to_the_agent():
    config = TextRunConfig(domain="retail", pre_verifier="v1")

    orchestrator = build_text_orchestrator(config, get_tasks()[0])

    assert orchestrator.agent.pre_verifier == "v1"


def test_builder_default_is_off():
    orchestrator = build_text_orchestrator(
        TextRunConfig(domain="retail"), get_tasks()[0]
    )

    assert orchestrator.agent.pre_verifier is None


def test_builder_rejects_agents_without_pre_verifier(environment, monkeypatch):
    monkeypatch.setattr(
        build_module.registry,
        "get_agent_factory",
        lambda name: lambda **kwargs: object(),
    )

    with pytest.raises(ValueError, match="pre-verifier"):
        build_agent("some_agent", environment, llm="test/agent", pre_verifier="v1")
