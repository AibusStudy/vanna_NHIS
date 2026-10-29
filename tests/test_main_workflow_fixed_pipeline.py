"""Workflow 고정 테스트 - 기존 흐름에 시각화 추가"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from vanna.core.llm import LlmResponse
from vanna.core.main_workflow.excutor import MainWorkflowExecutor
from vanna.core.main_workflow.state import MainWorkflowInput
from vanna.core.tool import ToolCall, ToolResult, ToolSchema
from vanna.core.user import User
from vanna.components import StatusCardComponent, TaskTrackerUpdateComponent
from vanna.integrations.openai import OpenAILlmService


class SqlLlm:
    def __init__(self, sql: str) -> None:
        self.sql = sql
        self.calls = 0
        self.requests = []

    async def send_request(self, request):
        self.calls += 1
        self.requests.append(request)
        return LlmResponse(
            tool_calls=[
                ToolCall(
                    id=f"sql_{self.calls}",
                    name="run_sql",
                    arguments={"sql": self.sql},
                )
            ]
        )


class RecordingRegistry:
    def __init__(self) -> None:
        self.names = []
        self.context_metadata = []

    async def execute(self, call, context):
        self.names.append(call.name)
        self.context_metadata.append(deepcopy(context.metadata))
        if call.name == "run_sql":
            return ToolResult(
                success=True,
                result_for_llm="ok",
                metadata={
                    "output_file": "query.csv",
                    "row_count": 2,
                    "columns": ["region", "sales"],
                },
            )
        return ToolResult(success=True, result_for_llm="visualized")


def workflow_input():
    user = User(id="user", group_memberships=["user"])
    return MainWorkflowInput(
        user_id="user",
        conversation_id="conversation",
        request_id="request",
        original_message="Show sales by region",
        system_prompt="You generate SQL.",
        tool_schemas=[
            ToolSchema(name="run_sql", description="Run SQL", parameters={}),
            ToolSchema(
                name="visualize_data",
                description="Visualize CSV",
                parameters={},
            ),
        ],
        tool_context=SimpleNamespace(user=user, metadata={}),
    )


@pytest.mark.asyncio
async def test_fixed_pipeline_executes_sql_then_visualization() -> None:
    registry = RecordingRegistry()
    llm = SqlLlm("SELECT region, SUM(sales) FROM orders GROUP BY region")
    executor = MainWorkflowExecutor(
        llm_service=llm,
        tool_registry=registry,
    )

    state = await executor.run(workflow_input())

    assert registry.names == ["run_sql", "visualize_data"]
    assert llm.requests[0].tool_choice == {
        "type": "function",
        "function": {"name": "run_sql"},
    }
    run_sql_context = registry.context_metadata[0]
    assert "main_workflow" in run_sql_context
    assert run_sql_context["main_workflow"]["stage"] == "sql_generation"
    assert "structured_question" in run_sql_context
    assert state.stage == "final"
    assert state.result["status"] == "success"
    assert state.subflow("visualization").status == "success"
    visualization_cards = [
        component.rich_component
        for component in state.ui_components
        if isinstance(component.rich_component, StatusCardComponent)
        and component.rich_component.title == "시각화 실행"
    ]
    visualization_tasks = [
        component.rich_component
        for component in state.ui_components
        if isinstance(component.rich_component, TaskTrackerUpdateComponent)
    ]
    assert [card.status for card in visualization_cards] == ["running", "success"]
    assert len(visualization_tasks) == 2


@pytest.mark.asyncio
async def test_fixed_pipeline_rejects_non_select_sql() -> None:
    registry = RecordingRegistry()
    llm = SqlLlm("DELETE FROM orders")
    executor = MainWorkflowExecutor(llm_service=llm, tool_registry=registry)

    state = await executor.run(workflow_input())

    assert registry.names == []
    assert llm.calls == 2
    assert state.result["status"] == "failed"
    assert "read-only SELECT" in state.result["message"]


def test_openai_payload_preserves_required_run_sql_tool_choice() -> None:
    service = object.__new__(OpenAILlmService)
    service.model = "test-model"
    request = workflow_input()
    llm_request = SimpleNamespace(
        system_prompt="Generate SQL",
        messages=[],
        tools=request.tool_schemas,
        tool_choice={"type": "function", "function": {"name": "run_sql"}},
        max_tokens=None,
    )

    payload = service._build_payload(llm_request)

    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "run_sql"},
    }
