from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from vanna.core.components import UiComponent

NodeStatus = Literal["success", "failed", "retry", "finish", "skipped"]
WorkflowStatus = Literal["success", "failed", "skipped"]


@dataclass(frozen=True)
class QuestUnderstand_Input:
    user_id: str
    conversation_id: str
    request_id: str
    original_message: str
    system_prompt: Optional[str]
    tool_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestUnderstand_RetryState:
    attempts_by_node: Dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str) -> int:
        next_count = self.attempts_by_node.get(node_id, 0) + 1
        self.attempts_by_node[node_id] = next_count
        return next_count

    def get_attempts(self, node_id: str) -> int:
        return self.attempts_by_node.get(node_id, 0)

    def to_dict(self) -> Dict[str, int]:
        return dict(self.attempts_by_node)


@dataclass
class QuestUnderstand_NodeResult:
    status: NodeStatus
    output: Any = None
    routing_intent: Optional[str] = None
    structured_question: Optional[Dict[str, Any]] = None
    debug_metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    failure_type: Optional[str] = None
    failure_detail: Optional[Dict[str, Any]] = None


@dataclass
class QuestUnderstand_State:
    """Mutable internal state used only while executing the workflow."""

    input: QuestUnderstand_Input
    routing_intent: Optional[str] = None
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    visited_nodes: List[str] = field(default_factory=list)
    last_node_result: Optional[QuestUnderstand_NodeResult] = None
    structured_question: Optional[Dict[str, Any]] = None
    debug_metadata: Dict[str, Any] = field(default_factory=dict)
    retry: QuestUnderstand_RetryState = field(default_factory=QuestUnderstand_RetryState)
    errors: List[str] = field(default_factory=list)
    failed_node_id: Optional[str] = None
    failure_type: Optional[str] = None
    failure_detail: Optional[Dict[str, Any]] = None

    @property
    def original_message(self) -> str:
        return self.input.original_message

    @property
    def retry_counts(self) -> Dict[str, int]:
        return self.retry.to_dict()

    def set_node_output(self, node_id: str, output: Any) -> None:
        self.node_outputs[node_id] = output

    def get_node_output(self, node_id: str) -> Any:
        return self.node_outputs.get(node_id)

    def add_error(self, error: str) -> None:
        if error:
            self.errors.append(error)


@dataclass(frozen=True)
class QuestUnderstand_FinalResult:
    status: WorkflowStatus
    intent: Optional[str] = None
    structured_output: Optional[Dict[str, Any]] = None
    ui_component: Optional["UiComponent"] = None
    errors: List[str] = field(default_factory=list)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    failed_node_id: Optional[str] = None
    failure_type: Optional[str] = None
    failure_detail: Optional[Dict[str, Any]] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "intent": self.intent,
            "structured_output": self.structured_output,
            "errors": list(self.errors),
            "retry_counts": dict(self.retry_counts),
            "failed_node_id": self.failed_node_id,
            "failure_type": self.failure_type,
            "failure_detail": self.failure_detail,
        }


def apply_node_result(
    state: QuestUnderstand_State,
    node_id: str,
    result: QuestUnderstand_NodeResult,
) -> QuestUnderstand_State:
    state.last_node_result = result

    if result.output is not None:
        state.set_node_output(node_id, result.output)
    if result.routing_intent is not None:
        state.routing_intent = result.routing_intent
    if result.structured_question is not None:
        state.structured_question = result.structured_question

    state.debug_metadata.update(result.debug_metadata)

    if result.error:
        state.add_error(result.error)
        state.failed_node_id = node_id

    if result.failure_type:
        state.failure_type = result.failure_type
        state.failed_node_id = node_id

    if result.failure_detail:
        state.failure_detail = result.failure_detail

    return state
