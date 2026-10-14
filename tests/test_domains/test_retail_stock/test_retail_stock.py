from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.domains.retail_stock.environment import get_environment, get_tasks
from tau2.evaluator.evaluator_action import ActionEvaluator
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator


def test_fixed_stock_lookup() -> None:
    environment = get_environment()
    result = environment.get_response(
        ToolCall(
            id="stock-check",
            name="get_variant_stock",
            arguments={"product_id": "9523456873", "item_id": "9612497925"},
        )
    )
    assert '"stock_quantity":8' in result.content.replace(" ", "")


def test_task_scores_answer_and_query_record() -> None:
    task = get_tasks()[0]
    trajectory = [
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="stock-check",
                    name="get_variant_stock",
                    arguments={
                        "product_id": "9523456873",
                        "item_id": "9612497925",
                    },
                )
            ],
        ),
        AssistantMessage(
            role="assistant",
            content="There are 8 available, so that is not enough; you are short 2.",
        ),
    ]
    assert ActionEvaluator.calculate_reward(task, trajectory).reward == 1.0
    assert CommunicateEvaluator.calculate_reward(task, trajectory).reward == 1.0


def test_forbidden_claim_fails_communication_score() -> None:
    task = get_tasks()[0]
    trajectory = [
        AssistantMessage(
            role="assistant",
            content=(
                "There are 8, which is not enough for 10; you are short 2. "
                "I have reserved them."
            ),
        )
    ]
    assert CommunicateEvaluator.calculate_reward(task, trajectory).reward == 0.0


def test_only_have_wording_passes_communication_score() -> None:
    task = get_tasks()[0]
    trajectory = [
        AssistantMessage(
            role="assistant",
            content="We only have 8 units in stock, which is 2 units short of 10.",
        )
    ]
    assert CommunicateEvaluator.calculate_reward(task, trajectory).reward == 1.0


def test_stock_changes_after_first_lookup_and_is_auditable() -> None:
    environment = get_environment()
    arguments = {"product_id": "9523456873", "item_id": "9612497925"}

    first = environment.tools.get_variant_stock(**arguments)
    second = environment.tools.get_variant_stock(**arguments)

    assert (first.stock_quantity, first.stock_version, first.simulation_time) == (
        8,
        1,
        0,
    )
    assert (second.stock_quantity, second.stock_version, second.simulation_time) == (
        2,
        2,
        1,
    )
    assert [event.model_dump() for event in second.events_since_previous_query] == [
        {
            "event_id": "background-purchase-after-first-query",
            "event_type": "background_purchase",
            "product_id": "9523456873",
            "item_id": "9612497925",
            "quantity_delta": -6,
            "stock_before": 8,
            "stock_after": 2,
            "version_before": 1,
            "version_after": 2,
            "simulation_time": 1,
        }
    ]


def test_phase_two_scores_revalidation_and_final_grounded_answer_separately() -> None:
    task = next(task for task in get_tasks() if task.id == "stock_changes_after_lookup")
    arguments = {"product_id": "9523456873", "item_id": "9612497925"}
    trajectory = [
        UserMessage(role="user", content="How many are available?"),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="first", name="get_variant_stock", arguments=arguments)
            ],
        ),
        ToolMessage(
            id="first",
            role="tool",
            content='{"stock_quantity": 8, "stock_version": 1}',
        ),
        AssistantMessage(role="assistant", content="There are 8 available."),
        UserMessage(role="user", content="Can you confirm you can supply all 5 now?"),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="second", name="get_variant_stock", arguments=arguments)
            ],
        ),
        ToolMessage(
            id="second",
            role="tool",
            content='{"stock_quantity": 2, "stock_version": 2}',
        ),
        AssistantMessage(
            role="assistant",
            content="We only have 2 now, so we cannot supply all 5; the shortfall is 3.",
        ),
    ]

    assert ActionEvaluator.calculate_reward(task, trajectory).reward == 1.0
    assert CommunicateEvaluator.calculate_reward(task, trajectory).reward == 1.0

    without_revalidation = trajectory[:5] + trajectory[7:]
    assert ActionEvaluator.calculate_reward(task, without_revalidation).reward == 0.0
    # The final claim is not grounded in the latest recorded evidence.
    assert (
        CommunicateEvaluator.calculate_reward(task, without_revalidation).reward == 0.0
    )


def test_phase_two_rejects_stale_final_claim_even_if_old_count_was_correct() -> None:
    task = next(task for task in get_tasks() if task.id == "stock_changes_after_lookup")
    trajectory = [
        ToolMessage(id="latest", role="tool", content='{"stock_quantity": 2}'),
        AssistantMessage(
            role="assistant",
            content="The earlier check showed 8, so all 5 are available.",
        ),
    ]
    assert CommunicateEvaluator.calculate_reward(task, trajectory).reward == 0.0


def test_phase_two_revalidation_must_follow_confirmation_request() -> None:
    task = next(task for task in get_tasks() if task.id == "stock_changes_after_lookup")
    arguments = {"product_id": "9523456873", "item_id": "9612497925"}
    trajectory = [
        UserMessage(role="user", content="How many are available?"),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="first", name="get_variant_stock", arguments=arguments)
            ],
        ),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="too-early", name="get_variant_stock", arguments=arguments)
            ],
        ),
        UserMessage(role="user", content="Can you confirm supply now?"),
    ]
    assert ActionEvaluator.calculate_reward(task, trajectory).reward == 0.0


def test_phase_two_accepts_non_english_grounded_quantity_statement() -> None:
    task = next(task for task in get_tasks() if task.id == "stock_changes_after_lookup")
    trajectory = [
        ToolMessage(
            id="latest",
            role="tool",
            content='{"stock_quantity": 2, "stock_version": 2}',
        ),
        AssistantMessage(
            role="assistant",
            content="目前庫存為 2 件，您的需求是 5 件，因此還差 3 件。",
        ),
    ]
    assert CommunicateEvaluator.calculate_reward(task, trajectory).reward == 1.0


def test_phase_two_rejects_wrong_supply_claim_despite_current_count() -> None:
    task = next(task for task in get_tasks() if task.id == "stock_changes_after_lookup")
    trajectory = [
        ToolMessage(
            id="latest",
            role="tool",
            content='{"stock_quantity": 2, "stock_version": 2}',
        ),
        AssistantMessage(
            role="assistant",
            content=(
                "There are 2 in stock for the requested 5, a shortfall of 3, "
                "but all 5 are available."
            ),
        ),
    ]
    assert CommunicateEvaluator.calculate_reward(task, trajectory).reward == 0.0


def test_phase_two_rejects_false_reservation_claim() -> None:
    task = next(task for task in get_tasks() if task.id == "stock_changes_after_lookup")
    trajectory = [
        ToolMessage(
            id="latest",
            role="tool",
            content='{"stock_quantity": 2, "stock_version": 2}',
        ),
        AssistantMessage(
            role="assistant",
            content=(
                "Only 2 are available for the requested 5, a shortfall of 3. "
                "I reserved them for you."
            ),
        ),
    ]
    assert CommunicateEvaluator.calculate_reward(task, trajectory).reward == 0.0


def test_phase_two_allows_explicit_reservation_denial() -> None:
    task = next(task for task in get_tasks() if task.id == "stock_changes_after_lookup")
    trajectory = [
        ToolMessage(
            id="latest",
            role="tool",
            content='{"stock_quantity": 2, "stock_version": 2}',
        ),
        AssistantMessage(
            role="assistant",
            content=(
                "There are 2 available for the requested 5, a shortfall of 3. "
                "Nothing has been reserved."
            ),
        ),
    ]
    assert CommunicateEvaluator.calculate_reward(task, trajectory).reward == 1.0
