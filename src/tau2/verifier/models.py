"""Data models for online verifier decisions and traces."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VerifierDecision(str, Enum):
    """Action the orchestrator should take for an agent proposal."""

    PASS = "pass"
    REVISE = "revise"
    BLOCK = "block"


class VerifierVerdict(BaseModel):
    """Structured verifier response for one proposed agent output."""

    decision: VerifierDecision
    reason: str
    revision_instruction: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class VerifierRecord(BaseModel):
    """Trace entry for one verifier invocation."""

    proposal_index: int
    history_length: int
    proposed_message: dict
    verdict: VerifierVerdict
    revision_attempt: int = 0
