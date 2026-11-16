# Midterm Project: Online Post-Verifier for Tau-Bench

## Overview

This project adds an online post-verifier to the Tau-Bench customer-service
agent framework. Before an agent response or tool call is committed, a separate
LLM verifier checks the proposed output against the domain policy, relevant
conversation, and a trusted event log built from actual environment tool
results. The verifier can pass the proposal, request one revision, or block it.

The trusted event log is created by the orchestrator rather than by the agent.
It pairs each committed tool call with its real environment result, so the
verifier does not have to trust an agent-written summary of database state.

## Main Files

Keep the following paths when placing the Python files in the submission zip.

| File | Purpose |
| --- | --- |
| `src/tau2/verifier/base.py` | Defines the abstract verifier interface. |
| `src/tau2/verifier/models.py` | Defines structured `pass`, `revise`, and `block` verdicts and verifier trace records. |
| `src/tau2/verifier/llm_verifier.py` | Implements the LLM post-verifier, its prompt, structured context, optional read-only fallback, and verdict parsing. |
| `src/tau2/verifier/__init__.py` | Exports the verifier classes and data models. |
| `src/tau2/orchestrator/orchestrator.py` | Intercepts uncommitted proposals, builds the trusted event log, invokes the verifier, and commits only approved proposals. |
| `src/tau2/agent/llm_agent.py` | Regenerates an uncommitted proposal using private verifier feedback when revision is requested. |
| `src/tau2/data_model/simulation.py` | Stores verifier configuration in the simulation run configuration. |
| `src/tau2/runner/build.py` | Constructs the verifier and injects it into the text orchestrator. |
| `src/tau2/cli.py` | Provides the verifier command-line options. |
| `tests/test_domains/test_retail_stock/test_online_verifier.py` | Tests approval, rejection, revision, trusted-event serialization, prompt isolation, and optional read-only checks. |

## Requirements

- Python 3.12
- `uv`
- Ollama
- The `qwen3.5:latest` Ollama model, or another LiteLLM-compatible model

Install the Python dependencies from the repository root:

```bash
uv sync --extra dev
```

Install and start the local model:

```bash
ollama pull qwen3.5:latest
ollama serve
```

## Running the Post-Verifier Experiment

The following command runs retail tasks 20, 41, 71, 72, 79, 82, and 98 twice
each. It uses the same local Qwen model for the customer-service agent, user
simulator, and post-verifier. No pre-verifier is enabled.

```bash
uv run tau2 run \
  --domain retail \
  --task-ids 20 41 71 72 79 82 98 \
  --agent-llm ollama_chat/qwen3.5:latest \
  --agent-llm-args '{"api_base":"http://127.0.0.1:11434","temperature":0,"num_ctx":4096,"think":false}' \
  --user-llm ollama_chat/qwen3.5:latest \
  --user-llm-args '{"api_base":"http://127.0.0.1:11434","temperature":0,"num_ctx":4096,"think":false}' \
  --verifier-model ollama_chat/qwen3.5:latest \
  --verifier-llm-args '{"api_base":"http://127.0.0.1:11434","temperature":0,"num_ctx":4096,"think":false}' \
  --verifier-max-revisions 1 \
  --num-trials 2 \
  --seed 300 \
  --max-concurrency 1 \
  --max-steps 200 \
  --timeout 7200 \
  --verbose-logs \
  --llm-log-mode all \
  --save-to retail-hard7-post-verifier-2trials \
  --auto-resume
```

Simulation results are written under `data/simulations/`. Each completed run
contains the normal conversation trajectory and separate verifier records with
the proposal, verdict, reason, confidence, and revision attempt.

## Running the Tests

```bash
uv run pytest tests/test_domains/test_retail_stock/test_online_verifier.py -q
```

The broader verifier and pre-verifier regression tests can be run with:

```bash
uv run pytest \
  tests/test_domains/test_retail_stock/test_online_verifier.py \
  tests/test_pre_verifier.py -q
```

## Design Notes

- A proposed response or tool call remains uncommitted until verification
  succeeds.
- Rejected proposals cannot execute environment tools or enter the official
  customer-visible trajectory.
- Conversation text is used to understand user intent and confirmation.
- The orchestrator-built trusted event log is the authoritative source for
  executed tool calls and returned system state.
- Hidden task instructions and evaluation criteria are not sent to the
  verifier.
- Read-only verifier tools are optional and are not required when the trusted
  event log already provides sufficient evidence.
- The verifier has no write capability.

## Current Limitation

Local LLM verification adds significant latency because every proposed agent
output requires an additional model call. Verifier prompts must also scope
per-message rules carefully; for example, “one tool call at a time” applies to
the current proposal, not to the total number of tool calls in the conversation
history.
