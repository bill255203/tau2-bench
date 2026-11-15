"""Verifier protocol for transactional agent-output gating."""

from abc import ABC, abstractmethod

from tau2.data_model.message import AssistantMessage, Message
from tau2.environment.tool import Tool
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
        tools: list[Tool] | None = None,
    ) -> VerifierVerdict:
        """Return a verdict without mutating the conversation or environment.

        The verifier gets the domain policy, committed history, proposal, and
        optionally isolated read-only tools. It must not receive the task, whose
        ``evaluation_criteria`` and ``user_scenario`` hold the expected actions
        and the user simulator's hidden instructions.
        """
        raise NotImplementedError
