# Online Verifier: Implementation and Local-Run Status

## Summary

This project adds an **online verifier** as a separate LLM call between a
customer-service agent's proposed output and the retail environment. An agent
can propose a customer reply or tool call, but that proposal is not committed
to the official trajectory and cannot execute against the environment until
the verifier approves it.

The implementation is validated by focused automated tests (**5 passed**) and
by a completed local-Ollama retail-stock simulation with an overall reward of
**1.0**. In that live run, the verifier detected an incorrect tool argument,
requested a private correction, and allowed only the corrected call to execute.

## What the Verifier Does

For each proposed agent output, the verifier receives:

- the applicable domain policy;
- the task specification and evaluation-relevant scenario information;
- the complete, already committed conversation history; and
- the new, still-uncommitted agent response or tool call.

It returns structured JSON with one of three decisions:

| Decision | Effect |
| --- | --- |
| `pass` | The proposal is committed to the trajectory and can be routed normally. A tool call may execute on the subsequent environment step. |
| `revise` | The standard LLM agent receives a private, temporary correction instruction and generates one replacement proposal. The original proposal remains uncommitted. |
| `block` | The proposal is discarded, the prior agent state is restored, and the simulation ends safely. No proposed tool call runs. |

This is a **fail-closed** design: if the verifier does not approve an output,
the proposed reply and any proposed tool call do not reach the official
conversation or the retail database.

## Implementation Map

| File | Responsibility |
| --- | --- |
| `src/tau2/orchestrator/orchestrator.py` | Holds proposals outside the official trajectory, calls the verifier, handles `pass`/`revise`/`block`, restores state after rejection, prints progress, and serializes verifier records into completed simulation results. |
| `src/tau2/verifier/llm_verifier.py` | Defines the separate LLM verifier prompt and sends policy, task, committed history, and proposed output to the verifier model. |
| `src/tau2/verifier/models.py` | Defines structured decisions and trace records, including `revision_attempt`. |
| `src/tau2/agent/llm_agent.py` | Adds private verifier-feedback regeneration for the standard LLM agent. The feedback and rejected proposal are not retained in customer-visible history. |
| `src/tau2/data_model/simulation.py` | Adds the `verifier_max_revisions` run configuration. |
| `src/tau2/cli.py` | Exposes `--verifier-max-revisions` from the command line. |
| `src/tau2/runner/build.py` | Builds the verifier and passes retry configuration to the orchestrator. |
| `tests/test_domains/test_retail_stock/test_online_verifier.py` | Provides deterministic tests for pass, block, revision-and-replacement, verifier construction, and structured verifier response parsing. |

## Live Local-Ollama Evidence

The completed live evaluation is saved at:

```text
data/simulations/retail-stock-local-verifier-revalidation-v6/
```

It ran the `stock_changes_after_lookup` task, which requires the agent to
perform an initial stock lookup and then revalidate inventory after the customer
asks whether the requested quantity can still be supplied. The simulation ended
normally with `user_stop` and achieved:

```text
Overall reward:       1.0
Action reward:        1.0
Communication reward: 1.0
Verifier records:     6
```

The first proposed stock lookup contained a real argument error: the agent used
the product ID `9523456873` as `item_id` instead of using item ID `9612497925`.
The verifier returned `revise` with 0.95 confidence and instructed the agent to
replace the incorrect `item_id`. The original proposal was not committed. The
agent then produced a corrected replacement, which the verifier approved on
revision attempt 1. Only that corrected lookup was allowed to execute.

The verifier subsequently approved the remaining policy-compliant steps:

1. report the initial availability of 8 units;
2. re-query stock after the customer asks for confirmation;
3. observe that a simulated background purchase reduced stock to 2 units;
4. report the final shortfall of 3 units without claiming a reservation or
   order; and
5. close the conversation normally.

This provides end-to-end evidence that the verifier can catch an incorrect
tool argument before execution, provide useful private revision feedback, and
allow the repaired multi-step task to finish successfully.

Focused tests also pass (`5 passed`), covering accepted proposals, blocked
proposals, revision-and-replacement behavior, verifier construction, and
structured verdict parsing.

## Professor-Ready Explanation

> We implemented an online verifier as a separate LLM safety gate between the
> customer-service agent and the retail environment. Before a customer reply or
> tool call is committed, the verifier inspects the applicable policy, task,
> committed conversation history, and proposed output. It can approve the
> action, block it safely, or request one private corrective re-generation from
> the agent. Rejected actions never execute or modify the environment. We also
> record every verifier decision, including initial and corrected attempts, so
> the experiment can audit when and why a proposed action was changed or
> stopped. In a completed local retail-stock run, the verifier caught an
> incorrect item identifier before it executed, supplied a private correction,
> and the corrected multi-step interaction received a reward of 1.0.

## Compact Verifier Context

The first local run showed that a naïve full message serialization can include
provider metadata and private `reasoning_content` stored in `raw_data`. This
made a simple verifier call unnecessarily large and slow. The verifier now
uses a compact representation of each history message. It retains only public,
decision-relevant evidence:

- participant role and visible text;
- tool-call IDs, names, arguments, and requestors; and
- tool-result IDs, content, requestors, and error status.

It deliberately excludes provider `raw_data`, private reasoning traces, token
usage, costs, timestamps, audio metadata, and other execution metadata. A
test confirms that a sentinel private reasoning string and `raw_data` fields
cannot reach the verifier prompt. This preserves the information required for
policy checking while avoiding context growth from internal provider traces.

## Limitation

The verifier is an LLM-based online judgment layer, not a formal proof of
long-horizon task success. It can assess whether the next action appears
grounded and policy-compliant from current evidence, but it cannot guarantee
that a multi-step conversation ultimately satisfies every hidden final
database state. The successful local run is one end-to-end case rather than a
statistical performance result; multiple tasks and trials are still required
for a broad empirical claim.
