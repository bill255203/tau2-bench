# Online Verifier Phase 1 Implementation Report

## Outcome

Phase 1 implements optional online verification for half-duplex text agents. When enabled, every generated customer-service agent output is treated as an uncommitted proposal. A separate verifier call receives the full committed trajectory, domain policy, task, and proposed output. Only a `pass` verdict commits the proposal and allows normal routing to the customer or environment.

The feature is disabled by default, so existing simulations retain their previous behavior.

## Implemented flow

```text
Customer message or Tool result
        ↓
Copy current agent state
        ↓
Generate proposal against the copied state
        ↓
Call verifier with policy + task + history + proposal
        ↓
Pass                         Revise / Block
  ↓                                ↓
Commit proposed state              Restore original state
Append proposal to trajectory      Do not append proposal
Route to customer/environment      Do not send or execute
```

In this phase, both `revise` and `block` fail closed and terminate the simulation with `verifier_rejected`. Automatic regeneration is intentionally deferred so Phase 1 can validate the transaction boundary before adding retry-loop complexity.

## Code changes

### Verifier package

Added `src/tau2/verifier/`:

- `base.py`: abstract verifier interface;
- `models.py`: `pass`, `revise`, and `block` verdict and trace models;
- `llm_verifier.py`: LLM-backed verifier using the existing LiteLLM generation path;
- `__init__.py`: public exports.

The LLM verifier requests a strict JSON result containing:

```json
{
  "decision": "pass | revise | block",
  "reason": "concise explanation",
  "revision_instruction": "specific correction or null",
  "confidence": 0.0
}
```

### Transactional orchestration

`Orchestrator` now accepts an optional `verifier`. When present, it:

1. copies the current agent state;
2. generates the proposal against another copy;
3. invokes the verifier before trajectory commit or tool execution;
4. commits the proposed state only on `pass`;
5. restores the original state and fails closed otherwise;
6. records every verifier invocation.

Verifier traces are written to `SimulationRun.info.verifier` and remain separate from the customer-visible conversation.

### Configuration and CLI

`TextRunConfig` now supports:

```python
verifier_model: str | None
verifier_llm_args: dict
```

The CLI supports:

```bash
--verifier-model MODEL
--verifier-llm-args '{"temperature": 0.0}'
```

`build_text_orchestrator()` creates and injects an `LLMVerifier` when a verifier model is configured.

Example:

```bash
tau2 run \
  --domain retail_stock \
  --agent-llm <agent-model> \
  --user-llm <user-model> \
  --verifier-model <verifier-model> \
  --verifier-llm-args '{"temperature": 0.0}' \
  --num-tasks 1 \
  --num-trials 1
```

## Retail-stock validation

Added `tests/test_domains/test_retail_stock/test_online_verifier.py` with four checks:

1. A passing verifier is called with the retail-stock policy, task, committed history, and proposed `get_variant_stock` call.
2. A passing verdict commits the copied agent state and proposal; normal routing then executes the stock tool.
3. A blocking verdict restores the original agent state, leaves the official trajectory unchanged, and does not execute the stock tool.
4. The text orchestrator builder enables the LLM verifier from configuration, and the LLM verifier parses a mocked structured result.

No external model request is required by these tests.

## Validation results

```text
Retail-stock verifier tests: 4 passed
Ruff lint: passed
Ruff format check: passed
```

The repository's existing API-backed orchestrator and LLM utility tests were also attempted, but they require an `OPENAI_API_KEY` and could not run in the current environment. Their failures were authentication failures, not verifier assertion failures.

## Current limitations

- Half-duplex text mode only;
- no automatic revision call yet;
- `revise` and `block` both fail closed;
- verifier token/cost is available in its generated response internally but is not yet aggregated into dedicated simulation-level verifier cost fields;
- confidence is traced but not calibrated or used as an execution threshold;
- no live provider call was made because the environment has no configured API key;
- full-duplex voice mode is unchanged.

## Recommended Phase 2

Add one bounded revision attempt:

1. restore the pre-proposal agent state;
2. add verifier feedback to agent-only context;
3. generate a corrected proposal;
4. verify it once more;
5. commit on `pass`, otherwise block safely.

Rejected proposals and verifier feedback must remain outside the official customer-visible trajectory.
