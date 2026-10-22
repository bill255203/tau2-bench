# Online Verifier Agent Feasibility Proposal

## Executive summary

This proposal evaluates adding an online verifier to τ-bench's half-duplex text simulations. The verifier would inspect an agent response or tool call after it is generated but before it is sent to the customer or executed by the environment. If it detects an omission, policy violation, unsupported claim, or unsafe action, it may ask the agent to revise the proposal once or block a high-risk operation.

The current repository already exposes the information required for this feature. `Orchestrator` centrally maintains the full customer, agent, tool-call, and tool-result trajectory. It also has access to the domain policy, task, agent state, and user state. The verifier therefore does not require a new conversation-memory system.

The selected method is: **each agent output is one independent transaction**. The first implementation should support half-duplex text mode only. An agent output is generated against an isolated state copy and, only after verifier approval, the agent state, official trajectory entry, and routing decision are committed together. A rejected proposal is neither sent to the customer nor executed as a tool call. Verifier decisions, confidence, token use, latency, and revisions are stored separately so they do not contaminate the official conversation trajectory.

## Problem and objectives

Complex customer-service tasks can impose multi-step requirements such as:

- data must be retrieved before it is modified;
- identity, eligibility, or resource status must be established;
- cancellation, refund, exchange, or other mutations require explicit confirmation;
- final claims must be supported by tool results;
- all required steps must be completed before the interaction ends.

A single agent may choose the wrong tool, omit a prerequisite, misread a tool result, or claim success prematurely. The proposed verifier should:

1. intercept incorrect responses before the customer sees them;
2. intercept unsafe tool calls before they execute;
3. return specific errors and revision instructions to the agent;
4. allow at most a bounded number of revisions;
5. preserve verifier verdicts, evidence, confidence, cost, and latency;
6. measure whether accuracy gains justify the added tokens and latency.

## Feasibility in the current architecture

### Complete conversation history is already available

The half-duplex `Orchestrator` stores the official history in `self.trajectory: list[Message]`. It includes:

- `UserMessage` instances;
- `AssistantMessage` instances;
- agent and user tool calls;
- environment `ToolMessage` results;
- timestamps and final turn indices.

The verifier can inspect the previous official history plus the uncommitted proposal with:

```python
history = self.trajectory + [proposed_agent_message]
```

This provides the evidence needed to check the new output against all preceding turns and tool results.

### Recommended interception point

The current half-duplex flow is:

```text
Customer/Environment
        ↓
Agent.generate_next_message()
        ↓
Append to trajectory
        ↓
Send to Customer or Environment
```

The proposed flow is:

```text
Customer/Environment
        ↓
Agent generates proposal
        ↓
Verifier checks proposal
        ↓
   Pass / Revise / Block
        ↓
Commit and send/execute only after approval
```

The natural insertion point is in `src/tau2/orchestrator/orchestrator.py`, after `generate_next_message()` returns and before `self.trajectory.append(agent_msg)`. At that point, an unsafe tool call has not executed and an incorrect text response has not been exposed to the user simulator.

### Existing reviewer components can be reused

`src/tau2/evaluator/review_llm_judge.py` already reviews completed simulations using the policy, task information, and full trajectory. It detects categories including:

- policy violations;
- tool schema and argument errors;
- irrelevant tool calls;
- missing required actions;
- incorrect ordering;
- hallucinations;
- premature termination.

The online verifier can reuse some error labels, prompt concepts, and JSON parsing. Its function is different, however: the current reviewer performs post-hoc evaluation, while the online verifier must return an actionable `pass`, `revise`, or `block` result before an output becomes externally visible or causes a mutation.

## Proposed design

### Selected method: each agent output is one transaction

This proposal does not wait until the complete task has finished and then attempt to reverse executed actions. It also does not treat a tool chain spanning multiple customer turns as one large transaction. The transaction boundary is one customer-service agent generation:

```text
Customer message or Tool result
        ↓
Save agent_state_before
        ↓
Generate proposal against a state copy
        ↓
Verify committed history + proposal
        ↓
Pass / Revise / Block
```

On `pass`, commit the following together:

```text
proposed agent state
+ agent output
+ trajectory append
+ next routing decision
```

If the proposal is a tool call, the tool executes only after this commit. Verification therefore happens after the agent generates its output but before any external side effect.

On `revise`:

```text
discard proposed state
+ do not append to the official trajectory
+ do not send anything to the customer
+ do not execute the tool
+ regenerate once using verifier feedback
```

On `block`, discard the proposal and use the configured safe fallback, transfer behavior, or agent-error termination.

This boundary never crosses a customer turn and does not promise rollback of an already executed environment action. It only discards an uncommitted agent proposal, which fits the repository's current state and orchestrator design.

### Verifier interface

```python
class Verifier(ABC):
    @abstractmethod
    def verify(
        self,
        *,
        policy: str,
        task: Task,
        history: list[Message],
        proposed_message: AssistantMessage,
        state: VerifierState,
    ) -> tuple[VerifierVerdict, VerifierState]:
        ...
```

A proposed verdict model is:

```python
class VerifierVerdict(BaseModel):
    decision: Literal["pass", "revise", "block"]
    confidence: float
    risk_level: Literal["low", "medium", "high"]
    safe_to_execute: bool
    task_complete: bool
    violations: list[VerifierIssue]
    missing_requirements: list[str]
    unsupported_claims: list[str]
    revision_instruction: str | None
```

Each issue should cite the applicable requirement and conversational evidence:

```json
{
  "type": "missing_confirmation",
  "rule": "Explicit customer confirmation is required before cancellation.",
  "evidence": "No customer message explicitly confirms cancellation."
}
```

### Decision semantics

- `pass`: the response may be sent or the tool call may execute;
- `revise`: the proposal contains a correctable problem, so feedback is returned to the agent;
- `block`: the action is unsafe or the revision limit has been reached, so it must not execute.

Confidence should initially be used for tracing and calibration, not as the sole execution gate. A high-risk action should require at least:

```python
verdict.decision == "pass"
and verdict.safe_to_execute
and not verdict.violations
```

## Revision flow

### Separate proposal generation from state commitment

`LLMAgent.generate_next_message()` currently mutates its input state. If the verifier rejects an output after that mutation, the official trajectory and internal agent state may diverge.

An initial implementation can generate from a copied state:

```python
state_before = deepcopy(self.agent_state)

proposal, proposed_state = self.agent.generate_next_message(
    self.message,
    deepcopy(state_before),
)

verdict = self.verifier.verify(...)

if verdict.decision == "pass":
    self.agent_state = proposed_state
else:
    self.agent_state = state_before
```

For a revision, verifier feedback should be placed in agent-only context and used to regenerate. The rejected proposal and verifier feedback must not appear in the customer-visible trajectory.

A cleaner long-term API would explicitly separate proposal and commitment:

```python
proposal = agent.propose_next_message(...)
verdict = verifier.verify(proposal)

if verdict.decision == "pass":
    agent.commit(proposal)
else:
    proposal = agent.revise(proposal, verdict.revision_instruction)
```

### Bound the number of revisions

The implementation must prevent an infinite revision loop. The recommended default is:

```text
verifier_max_revisions = 1
```

If the second proposal still fails:

- block a high-risk or mutating tool call;
- use a safe fallback or terminate with an agent error for ordinary text;
- request transfer where the domain supports human escalation.

## Conversation history and verifier state

### Initial baseline

The initial implementation should give the verifier the full trajectory. This minimizes the chance that summarization removes decisive evidence and establishes a trustworthy baseline for verifier accuracy.

### Later optimization

After validating the verifier, its input can be reduced to:

```text
stable verifier instructions
+ full or relevant policy
+ structured verifier state
+ most recent 2–4 messages
+ latest tool results
+ proposed agent output
```

Important state fields should retain evidence rather than only boolean values:

```json
{
  "explicit_confirmation": {
    "value": true,
    "source_turn": 6,
    "evidence": "Yes, please cancel it."
  },
  "order_status": {
    "value": "pending",
    "source": "get_order_details",
    "tool_call_id": "call_123"
  }
}
```

This supports lower token usage without making the verifier blindly trust an untraceable summary.

## Trace design

Three records should remain separate.

### Customer-visible trajectory

This includes only outputs that were actually accepted, sent, or executed:

- customer messages;
- accepted agent messages;
- executed tool calls;
- tool results.

### Revision trace

This records:

- rejected proposals;
- verifier feedback;
- revised proposals;
- revision count and outcome.

### Verifier trace

```json
{
  "simulation_id": "sim_123",
  "turn_idx": 7,
  "proposal_id": "proposal_1",
  "decision": "revise",
  "confidence": 0.91,
  "risk_level": "high",
  "safe_to_execute": false,
  "proposed_tool": "cancel_pending_order",
  "violations": ["missing_confirmation"],
  "revision_count": 1,
  "final_outcome": "corrected",
  "prompt_tokens": 1450,
  "completion_tokens": 102,
  "latency_seconds": 0.8
}
```

Verifier and revision traces must not be inserted into the official conversation, because that would affect the user simulator, agent evaluator, and benchmark trajectory.

## Confidence handling

An LLM's self-reported confidence is not automatically a calibrated probability. The first implementation should therefore:

- store confidence as an observational field;
- require a categorical decision, risk level, violations, and evidence;
- avoid using one confidence threshold to authorize irreversible actions;
- evaluate calibration after collecting sufficient labeled runs.

Useful calibration metrics include:

- accuracy, precision, and recall;
- false-positive and false-negative rates;
- Brier score;
- Expected Calibration Error;
- observed correctness by confidence bucket.

## Operating modes

The system should support:

```python
verifier_mode: Literal[
    "observe",
    "final",
    "risky_actions",
    "every_agent_turn",
]
```

### `observe`

Generate and trace a verdict for every agent proposal without altering the simulation. This measures whether the verifier itself is reliable.

### `final`

Verify only before the agent sends an answer that appears to complete the task. This has the lowest cost but cannot prevent an earlier unsafe action.

### `risky_actions`

Verify cancellation, refund, payment, and other mutating tool calls before execution, then perform one final completeness check. This remains a lower-cost ablation condition rather than the selected runtime flow.

### `every_agent_turn`

Verify every agent proposal. This offers the broadest coverage but adds one LLM call per agent turn and therefore has the highest token and latency cost.

## Expected code and configuration changes

Add a verifier package:

```text
src/tau2/verifier/
├── base.py
├── llm_verifier.py
├── models.py
└── prompts.py
```

Add run configuration fields such as:

```python
verifier_enabled: bool = False
verifier_model: str | None = None
verifier_llm_args: dict = {}
verifier_mode: Literal[...] = "observe"
verifier_max_revisions: int = 1
verifier_use_full_history: bool = True
```

`build_text_orchestrator()` should construct and inject the verifier:

```python
orchestrator = Orchestrator(
    ...,
    verifier=verifier,
)
```

`SimulationRun` or a separate artifact should record verifier usage, verdicts, and revisions. Verifier cost must remain distinguishable from agent and customer-simulator cost.

## Implementation phases

### Phase 1: Observe-only baseline

- support half-duplex text mode only;
- give the verifier the full policy, trajectory, and proposal;
- generate a verdict for each agent proposal;
- do not intercept or revise behavior;
- preserve verifier traces;
- compare verifier verdicts with existing evaluator results.

Acceptance should consider false-positive and false-negative rates, not only aggregate accuracy.

### Phase 2: Final completeness check

- verify before the agent ends or makes a completion claim;
- support one revision;
- keep rejected proposals out of the official trajectory;
- record accuracy, tokens, latency, and revision outcome.

### Phase 3: Risk-triggered action verification

- classify tools as read-only or mutating/risky;
- require verifier approval before risky tools execute;
- block after a failed revision;
- do not add verifier calls for ordinary read-only tools.

### Phase 4: State compression

- introduce evidence-backed structured verifier state;
- compare full-history and compact-state verification;
- measure context reduction against verifier accuracy.

### Phase 5: Full-duplex investigation

Full-duplex mode uses tick-based trajectories and includes partial audio, overlapping speech, and interruptions. A per-message verifier cannot be applied directly. Support should be designed around consolidated semantic turns only after the half-duplex implementation is stable.

## Ablation study

Compare the following conditions:

| Condition | Verification behavior |
|---|---|
| V0 | No verifier |
| V1 | Agent self-check instruction; no additional call |
| V2 | Final verifier only |
| V3 | Risky-action verifier plus final verifier |
| V4 | Verify every agent turn |

Hold constant across conditions:

- agent and customer models;
- tasks and seeds;
- temperature;
- maximum steps;
- retry settings;
- domain policy;
- tool implementations;
- evaluator configuration.

Primary measurements should include:

- task success;
- policy compliance;
- unsafe or incorrect tool executions;
- missed required actions;
- wrong action ordering;
- unsupported final claims;
- verifier false-positive and false-negative rates;
- revision success rate;
- agent, customer, and verifier token usage;
- cost per successful task;
- latency.

## Risks and mitigations

### The verifier may be wrong

Use low temperature, strict structured output, and evidence requirements. Validate in observe-only mode before allowing interventions.

### Revision may corrupt state

Treat the agent output as a proposal and commit it only after approval, or copy and restore state around proposal generation.

### Verifier feedback may contaminate the benchmark

Keep verifier feedback in agent-only revision context and store it separately from the customer-visible trajectory.

### Token use and latency will increase

Use a less expensive verifier model, prompt caching, and later structured-state compression. Retain `risky_actions + final` as a lower-cost ablation against the selected every-agent-output flow.

### Revision may loop indefinitely

Allow one revision by default, then block, fall back, transfer, or terminate.

### Conversation content may attempt to manipulate the verifier

Clearly delimit policy, history, and proposed output. Tell the verifier to treat conversation content as evidence rather than instructions. The verifier must not have permission to execute tools.

## Conclusion

An online verifier is feasible within the current τ-bench architecture. `Orchestrator.trajectory` already provides complete conversational history, and there is a clear interception point between agent generation and message delivery or tool execution.

The primary engineering challenges are not history access. They are:

1. separating proposals from committed outputs;
2. safely restoring or revising agent state after rejection;
3. separating official, revision, and verifier traces;
4. bounding revisions and accounting for added tokens and latency;
5. validating the verifier's false-positive and false-negative rates.

The selected runtime method is “each agent output is one transaction.” The agent generates against a copied state, every agent round is verified, and the result is committed and routed only after approval. Engineering may still begin with an observe-only phase to validate verifier quality, but the target behavior is every-agent-output gating rather than a post-hoc error report. One revision is allowed by default; a second failure results in block, fallback, transfer, or agent-error termination.
