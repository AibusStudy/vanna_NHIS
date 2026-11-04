"""Graph and default nodes for Data-Discovering workflow."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Optional, Set

from .edge import EdgeCondition, DataDiscover_Edge
from .node import DataDiscover_Node
from .state import DataDiscover_NodeResult, DataDiscover_State

logger = logging.getLogger(__name__)


class WorkflowGraphError(ValueError):
    """Raised when a Data-Discovering workflow graph is invalid."""


class WorkflowGraph:
    """Node-edge graph executed by DataDiscoverSubWorkflowExecutor."""

    def __init__(self) -> None:
        self._nodes: Dict[str, DataDiscover_Node] = {}
        self._edges_by_source: DefaultDict[str, List[DataDiscover_Edge]] = defaultdict(list)
        self.start_node_id: Optional[str] = None
        self.end_node_ids: Set[str] = set()

    @property
    def node_ids(self) -> Set[str]:
        return set(self._nodes.keys())

    def add_node(
        self,
        node: DataDiscover_Node,
        *,
        start: bool = False,
        end: bool = False,
    ) -> "WorkflowGraph":
        if node.node_id in self._nodes:
            raise WorkflowGraphError(f"Duplicate workflow node: {node.node_id}")

        self._nodes[node.node_id] = node
        if start:
            self.set_start(node.node_id)
        if end:
            self.add_end(node.node_id)
        return self

    def set_start(self, node_id: str) -> "WorkflowGraph":
        self._ensure_node_exists(node_id)
        self.start_node_id = node_id
        return self

    def add_end(self, node_id: str) -> "WorkflowGraph":
        self._ensure_node_exists(node_id)
        self.end_node_ids.add(node_id)
        return self

    def add_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        condition: Optional[EdgeCondition] = None,
        label: Optional[str] = None,
    ) -> "WorkflowGraph":
        self._ensure_node_exists(source_node_id)
        self._ensure_node_exists(target_node_id)
        self._edges_by_source[source_node_id].append(
            DataDiscover_Edge(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                condition=condition,
                label=label,
            )
        )
        return self

    def get_node(self, node_id: str) -> DataDiscover_Node:
        self._ensure_node_exists(node_id)
        return self._nodes[node_id]

    def get_edges(self, source_node_id: str) -> List[DataDiscover_Edge]:
        self._ensure_node_exists(source_node_id)
        return list(self._edges_by_source.get(source_node_id, []))

    def validate(self) -> None:
        if not self._nodes:
            raise WorkflowGraphError("Workflow graph has no nodes.")
        if self.start_node_id is None:
            raise WorkflowGraphError("Workflow graph has no start node.")
        self._ensure_node_exists(self.start_node_id)
        if not self.end_node_ids:
            raise WorkflowGraphError("Workflow graph has no end nodes.")

        for end_node_id in self.end_node_ids:
            self._ensure_node_exists(end_node_id)
        for source_node_id, edges in self._edges_by_source.items():
            self._ensure_node_exists(source_node_id)
            for edge in edges:
                self._ensure_node_exists(edge.target_node_id)
        for node_id in self._nodes:
            if node_id not in self.end_node_ids and not self.get_edges(node_id):
                raise WorkflowGraphError(
                    f"Non-end workflow node has no outgoing edges: {node_id}"
                )

        unreachable_node_ids = self._find_unreachable_nodes()
        if unreachable_node_ids:
            raise WorkflowGraphError(
                "Workflow graph has unreachable nodes: "
                + ", ".join(sorted(unreachable_node_ids))
            )

    def _find_unreachable_nodes(self) -> Set[str]:
        if self.start_node_id is None:
            return self.node_ids

        visited_node_ids: Set[str] = set()
        queue = deque([self.start_node_id])
        while queue:
            node_id = queue.popleft()
            if node_id in visited_node_ids:
                continue
            visited_node_ids.add(node_id)
            for edge in self.get_edges(node_id):
                if edge.target_node_id not in visited_node_ids:
                    queue.append(edge.target_node_id)
        return self.node_ids - visited_node_ids

    def _ensure_node_exists(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise WorkflowGraphError(f"Unknown workflow node: {node_id}")


class SearchPlanCondition:
    def __init__(self, *, exists: bool) -> None:
        self.exists = exists

    async def evaluate(
        self,
        state: DataDiscover_State,
        last_node_result: DataDiscover_NodeResult,
    ) -> bool:
        structured_output = state.input.structured_output or {}
        return _has_search_plan_queries(structured_output) is self.exists


class DataDiscoveryRouterNode:
    """Routes SQL discovery to batch metadata search or agentic metadata search."""

    node_id = "data_discovery_router"

    async def run(self, state: DataDiscover_State) -> DataDiscover_NodeResult:
        if state.input.status != "success":
            return DataDiscover_NodeResult(
                status="skipped",
                warnings=["data discovery skipped: question_understanding not successful"],
            )
        if state.input.intent != "sql":
            return DataDiscover_NodeResult(
                status="skipped",
                warnings=["data discovery skipped: intent is not sql"],
            )
        return DataDiscover_NodeResult(status="success")


class MetadataSearchNode:
    """Runs batch metadata search from question-understanding search_plan."""

    node_id = "metadata_search"

    def __init__(self, search_service: Any) -> None:
        self.search_service = search_service

    async def run(self, state: DataDiscover_State) -> DataDiscover_NodeResult:
        structured_output = state.input.structured_output or {}
        queries = self._build_queries(structured_output)

        if not queries:
            return DataDiscover_NodeResult(
                status="failed",
                error="metadata_search_plan_invalid: search_plan.queries is missing or empty",
                failure_type="metadata_search_plan_invalid",
                failure_detail={"structured_question": structured_output},
            )

        try:
            result = self.search_service.search_batch(queries)
        except Exception as exc:
            return DataDiscover_NodeResult(
                status="failed",
                error=f"metadata_execution_error: {type(exc).__name__}: {exc}",
                failure_type="metadata_execution_error",
                failure_detail={
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "queries": queries,
                },
            )

        result_dict = dict(result)
        if _is_metadata_semantic_mismatch(result_dict):
            return DataDiscover_NodeResult(
                status="failed",
                metadata_output={"status": "semantic_mismatch", **result_dict},
                warnings=list(result_dict.get("warnings", [])),
                error="metadata_semantic_mismatch: metadata result does not match structured question",
                failure_type="metadata_semantic_mismatch",
                failure_detail={
                    "queries": queries,
                    "structured_question": structured_output,
                    "metadata_status": result_dict.get("status"),
                    "mismatch_reason": result_dict.get("mismatch_reason"),
                },
            )

        return DataDiscover_NodeResult(
            status="success",
            metadata_output={"status": "success", **result_dict},
            warnings=list(result_dict.get("warnings", [])),
        )

    @staticmethod
    def _build_queries(structured_output: dict[str, Any]) -> list[dict[str, Any]]:
        search_plan = structured_output.get("search_plan")
        if isinstance(search_plan, dict):
            raw_queries = search_plan.get("queries") or search_plan.get("search_queries")
        else:
            raw_queries = None

        if not isinstance(raw_queries, list):
            return []

        queries: list[dict[str, Any]] = []
        for index, raw_query in enumerate(raw_queries[:10], start=1):
            if not isinstance(raw_query, dict):
                continue
            query_text = raw_query.get("query")
            if not isinstance(query_text, str) or not query_text.strip():
                continue
            queries.append(
                {
                    "query_id": raw_query.get("query_id") or f"Q{index}",
                    "query": query_text.strip(),
                    "scope": raw_query.get("scope", "auto"),
                    "table_names": raw_query.get("table_names"),
                }
            )
        return queries


class AgenticMetadataSearchNode:
    """Runs original-question metadata search when no batch search_plan exists."""

    node_id = "agentic_metadata_search"

    def __init__(self, search_service: Any) -> None:
        self.search_service = search_service

    async def run(self, state: DataDiscover_State) -> DataDiscover_NodeResult:
        structured_output = state.input.structured_output or {}
        question = _question_from_structured_output(structured_output)
        if not question:
            return DataDiscover_NodeResult(
                status="success",
                metadata_output={"status": "skipped", "tables": [], "columns": []},
                warnings=["agentic metadata search skipped: question missing"],
            )

        queries = [
            {
                "query_id": "AQ1",
                "query": question,
                "scope": "auto",
            }
        ]

        try:
            result = self.search_service.search_batch(queries)
        except Exception as exc:
            return DataDiscover_NodeResult(
                status="failed",
                error=f"metadata_execution_error: {type(exc).__name__}: {exc}",
                failure_type="metadata_execution_error",
                failure_detail={
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "queries": queries,
                    "search_mode": "agentic_metadata_search",
                },
            )

        result_dict = dict(result)
        if _is_metadata_semantic_mismatch(result_dict):
            return DataDiscover_NodeResult(
                status="success",
                metadata_output={
                    **result_dict,
                    "status": "warning",
                    "warning_type": "agentic_metadata_semantic_mismatch",
                    "search_mode": "agentic_metadata_search",
                },
                warnings=[
                    *list(result_dict.get("warnings", [])),
                    "agentic_metadata_semantic_mismatch: metadata result does not match structured question; continuing to fewshot_search",
                ],
            )

        return DataDiscover_NodeResult(
            status="success",
            metadata_output={
                "status": "success",
                "search_mode": "agentic_metadata_search",
                **result_dict,
            },
            warnings=list(result_dict.get("warnings", [])),
        )


class FewShotSearchNode:
    """Runs few-shot search as a node while leaving tool usage available elsewhere."""

    node_id = "fewshot_search"

    def __init__(self, agent_memory: Any = None, *, limit: int = 3) -> None:
        self.agent_memory = agent_memory
        self.limit = limit

    async def run(self, state: DataDiscover_State) -> DataDiscover_NodeResult:
        if self.agent_memory is None:
            return DataDiscover_NodeResult(
                status="finish",
                fewshot_output={"status": "skipped", "examples": []},
                warnings=["fewshot search skipped: agent_memory missing"],
            )

        question = _question_from_structured_output(state.input.structured_output or {})
        if not question:
            return DataDiscover_NodeResult(
                status="finish",
                fewshot_output={"status": "skipped", "examples": []},
                warnings=["fewshot search skipped: question missing"],
            )

        try:
            results = await self.agent_memory.search_similar_usage(
                question=question,
                context=None,
                limit=self.limit,
                similarity_threshold=0.7,
            )
        except Exception as exc:
            return DataDiscover_NodeResult(
                status="finish",
                fewshot_output={"status": "failed", "examples": []},
                warnings=[f"fewshot search failed: {type(exc).__name__}: {exc}"],
            )

        examples: list[dict[str, Any]] = []
        for result in results[: self.limit]:
            memory = getattr(result, "memory", result)
            examples.append(
                {
                    "similarity": getattr(result, "similarity_score", None),
                    "question": getattr(memory, "question", None),
                    "sql": getattr(memory, "sql", None),
                    "tool_name": getattr(memory, "tool_name", None),
                    "args": getattr(memory, "args", None),
                }
            )

        return DataDiscover_NodeResult(
            status="finish",
            fewshot_output={"status": "success", "examples": examples},
        )


def build_data_discovering_graph(
    *,
    metadata_search_service: Any,
    agent_memory: Any = None,
    fewshot_limit: int = 3,
) -> WorkflowGraph:
    graph = WorkflowGraph()
    graph.add_node(
        DataDiscoveryRouterNode(),
        start=True,
    )
    graph.add_node(MetadataSearchNode(metadata_search_service))
    graph.add_node(AgenticMetadataSearchNode(metadata_search_service))
    graph.add_node(
        FewShotSearchNode(agent_memory, limit=fewshot_limit),
        end=True,
    )
    graph.add_edge(
        "data_discovery_router",
        "metadata_search",
        condition=SearchPlanCondition(exists=True),
        label="search_plan_exists",
    )
    graph.add_edge(
        "data_discovery_router",
        "agentic_metadata_search",
        condition=SearchPlanCondition(exists=False),
        label="search_plan_missing",
    )
    graph.add_edge("metadata_search", "fewshot_search", label="metadata_done")
    graph.add_edge("agentic_metadata_search", "fewshot_search", label="agentic_metadata_done")
    return graph


def build_data_discovering_subworkflow_executor(
    *,
    metadata_search_service: Any,
    agent_memory: Any = None,
    fewshot_limit: int = 3,
    max_steps: int = 5,
    retry_limit: int = 1,
) -> "DataDiscoverSubWorkflowExecutor":
    from .executor import DataDiscoverSubWorkflowExecutor

    return DataDiscoverSubWorkflowExecutor(
        build_data_discovering_graph(
            metadata_search_service=metadata_search_service,
            agent_memory=agent_memory,
            fewshot_limit=fewshot_limit,
        ),
        max_steps=max_steps,
        retry_limit=retry_limit,
    )


def _question_from_structured_output(structured_output: dict[str, Any]) -> str:
    for key in (
        "metadata_discovery_question",
        "original_question",
        "question",
        "user_question",
        "rewritten_question",
    ):
        value = structured_output.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _has_search_plan_queries(structured_output: dict[str, Any]) -> bool:
    search_plan = structured_output.get("search_plan")
    if not isinstance(search_plan, dict):
        return False
    raw_queries = search_plan.get("queries") or search_plan.get("search_queries")
    if not isinstance(raw_queries, list):
        return False
    return any(
        isinstance(query, dict)
        and isinstance(query.get("query"), str)
        and bool(query.get("query", "").strip())
        for query in raw_queries
    )







def _is_metadata_semantic_mismatch(result: dict[str, Any]) -> bool:
    if result.get("failure_type") == "metadata_semantic_mismatch":
        return True
    if result.get("metadata_semantic_mismatch") is True:
        return True
    if result.get("semantic_mismatch") is True:
        return True
    status = result.get("status")
    return status in {"metadata_semantic_mismatch", "semantic_mismatch", "mismatch"}


