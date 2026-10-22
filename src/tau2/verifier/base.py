"""Verifier protocol for transactional agent-output gating."""

from abc import ABC, abstractmethod

from tau2.data_model.message import AssistantMessage, Message
from tau2.data_model.tasks import Task
from tau2.verifier.models import VerifierVerdict


class Verifier(ABC):
    """Validate an uncommitted agent proposal against its conversation context."""

    @abstractmethod
    def verify(
        self,
        *,
        policy: str,
        task: Task,
        history: list[Message],
        proposed_message: AssistantMessage,
    ) -> VerifierVerdict:
        """Return a verdict without mutating the conversation or environment."""
        raise NotImplementedError
