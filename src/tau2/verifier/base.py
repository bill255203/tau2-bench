"""Verifier protocol for transactional agent-output gating."""

from abc import ABC, abstractmethod

from tau2.data_model.message import AssistantMessage, Message
from tau2.verifier.models import VerifierVerdict


class Verifier(ABC):
    """Validate an uncommitted agent proposal against its conversation context."""

    @abstractmethod
    def verify(
        self,
        *,
        policy: str,
        history: list[Message],
        proposed_message: AssistantMessage,
    ) -> VerifierVerdict:
        """Return a verdict without mutating the conversation or environment.

        The verifier gets only the domain policy, the committed history, and
        the proposal. It must not receive the task, whose
        ``evaluation_criteria`` and ``user_scenario`` hold the expected actions
        and the user simulator's hidden instructions.
        """
        raise NotImplementedError
