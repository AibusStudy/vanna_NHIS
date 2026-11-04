from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from vanna.core.components import UiComponent

NodeStatus = Literal["success", "failed", "retry", "finish", "skipped"]
WorkflowStatus = Literal["success", "failed", "skipped"]


@dataclass(frozen=True)
class DataDiscover_Input:
    status: str
    intent: Optional[str] = None
    structured_output: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    retry_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class DataDiscover_RetryState:
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
class DataDiscover_NodeResult:
    status: NodeStatus
    output: Any = None
    metadata_output: Optional[Dict[str, Any]] = None
    fewshot_output: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    failure_type: Optional[str] = None
    failure_detail: Optional[Dict[str, Any]] = None


@dataclass
class DataDiscover_State:
    input: DataDiscover_Input
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata_output: Optional[Dict[str, Any]] = None
    fewshot_output: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    retry: DataDiscover_RetryState = field(default_factory=DataDiscover_RetryState)
    last_node_result: Optional[DataDiscover_NodeResult] = None
    failed_node_id: Optional[str] = None
    failure_type: Optional[str] = None
    failure_detail: Optional[Dict[str, Any]] = None

    @property
    def retry_counts(self) -> Dict[str, int]:
        return self.retry.to_dict()

    def set_node_output(self, node_id: str, output: Any) -> None:
        self.node_outputs[node_id] = output

    def add_warning(self, warning: str) -> None:
        if warning:
            self.warnings.append(warning)

    def add_error(self, error: str) -> None:
        if error:
            self.errors.append(error)


@dataclass(frozen=True)
class DataDiscover_FinalResult:
    status: WorkflowStatus
    metadata_output: Optional[Dict[str, Any]] = None
    fewshot_output: Optional[Dict[str, Any]] = None
    ui_component: Optional["UiComponent"] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    failed_node_id: Optional[str] = None
    failure_type: Optional[str] = None
    failure_detail: Optional[Dict[str, Any]] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "metadata_output": self.metadata_output,
            "fewshot_output": self.fewshot_output,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "retry_counts": dict(self.retry_counts),
            "failed_node_id": self.failed_node_id,
            "failure_type": self.failure_type,
            "failure_detail": self.failure_detail,
        }


def apply_node_result(
    state: DataDiscover_State,
    node_id: str,
    result: DataDiscover_NodeResult,
) -> DataDiscover_State:
    state.last_node_result = result

    if result.output is not None:
        state.set_node_output(node_id, result.output)
    if result.metadata_output is not None:
        state.metadata_output = result.metadata_output
    if result.fewshot_output is not None:
        state.fewshot_output = result.fewshot_output

    for warning in result.warnings:
        state.add_warning(warning)

    if result.error:
        state.add_error(result.error)
        state.failed_node_id = node_id

    if result.failure_type:
        state.failure_type = result.failure_type
        state.failed_node_id = node_id

    if result.failure_detail:
        state.failure_detail = result.failure_detail

    return state
