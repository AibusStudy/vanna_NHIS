"""MainWorkflow executor for turn state orchestration."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from vanna.core.llm import LlmMessage, LlmRequest
from vanna.core.tool import ToolCall
from vanna.capabilities.sql_runner.statement import get_statement_type
from vanna.components import (
    ButtonGroupComponent,
    StatusCardComponent,
    Task,
    TaskTrackerUpdateComponent,
    UiComponent,
)

from vanna.core.question_understanding_subworkflow import (
    QuestUnderstand_Input,
    QuestionUnderstandSubWorkflowExecutor,
)

try:
    from vanna.core.data_discovering_subworkflow import (
        DataDiscover_Input,
        DataDiscoverSubWorkflowExecutor,
    )
except ImportError:
    DataDiscover_Input = None

from .state import CsvRequestState, MainWorkflowInput, MainWorkflowTurnState

logger = logging.getLogger(__name__)

FB1_FAILURE_TYPES = {
    "intent_classification_failed",
    "time_normalization_failed",
    "time_metadata_validation_failed",
    "question_structuring_failed",
    "search_queries_generation_failed",
}

JSON_RETRY_FAILURE_TYPES = {
    "json_validation_failed",
    "json_validation_retry_exceeded",
}

FB2_DATA_DISCOVERY_FAILURE_TYPES = {"metadata_execution_error"}
FB2_QUESTION_UNDERSTANDING_FAILURE_TYPES = {"metadata_semantic_mismatch"}


def _find_turnstate_log_dir() -> Path:
    env_path = os.getenv("GENSQL_TURNSTATE_LOG_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()

    search_starts = [Path.cwd(), Path(__file__).resolve()]
    for start in search_starts:
        current = start if start.is_dir() else start.parent
        for parent in [current, *current.parents]:
            gensql_dir = parent / "gensql"
            if gensql_dir.is_dir():
                return gensql_dir / "scripts" / "turnstate"

    return Path.cwd() / "gensql" / "scripts" / "turnstate"


TURNSTATE_LOG_DIR = _find_turnstate_log_dir()


def _safe_filename_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def _write_turn_state_file(event: str, state: MainWorkflowTurnState) -> Path | None:
    try:
        TURNSTATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        event_name = _safe_filename_part(event)[:80]
        turn_id = _safe_filename_part(state.turn_id)[:80]
        path = TURNSTATE_LOG_DIR / f"{timestamp}_{event_name}_{turn_id}.json"
        payload = {
            "event": event,
            "timestamp": timestamp,
            "turn_state": state.to_metadata(),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
    except Exception:
        logger.exception("[main_workflow.turn_state] failed_to_write_file event=%s", event)
        return None

# intent == sql
def _log_turn_state(event: str, state: MainWorkflowTurnState) -> None:
    file_path = _write_turn_state_file(event, state)
    logger.info(
        "[main_workflow.turn_state] %s turn_id=%s stage=%s operation=%s "
        "intent=%s question_understanding=%s data_discovery=%s sql_processing=%s file=%s",
        event,
        state.turn_id,
        state.stage,
        state.operation,
        state.structured_question.get("intent"),
        state.subflow("question_understanding").status,
        state.subflow("data_discovery").status,
        state.subflow("sql_processing").status,
        str(file_path) if file_path is not None else None,
    )

# intent != sql
def _log_turn_state_summary(event: str, state: MainWorkflowTurnState) -> None:
    file_path = _write_turn_state_file(event, state)
    logger.info(
        "[main_workflow.turn_state] %s turn_id=%s stage=%s operation=%s "
        "intent=%s question_understanding=%s data_discovery=%s file=%s",
        event,
        state.turn_id,
        state.stage,
        state.operation,
        state.structured_question.get("intent"),
        state.subflow("question_understanding").status,
        state.subflow("data_discovery").status,
        str(file_path) if file_path is not None else None,
    )


class MainWorkflowExecutor:
    """Runs Main workflows and records normalized turn state."""

    def __init__(
        self,
        question_understanding_executor: Optional[QuestionUnderstandSubWorkflowExecutor] = None,
        data_discovery_executor: Optional[DataDiscoverSubWorkflowExecutor] = None,
        router=None,
        csv_request_detector: Callable[[str], dict[str, Any]] | None = None,
        llm_service: Any = None,
        tool_registry: Any = None,
        llm_context_enhancer: Any = None,

        # steps, limit 적용x
        # max_workflow_steps, workflow_retry_limit는 현재 MainWorkflowExecutor에서 적용하지 않는다.
        # max_workflow_steps: int = 10,
        # workflow_retry_limit: int = 1,
        **_: Any,
    ):

        self.question_understanding_executor = question_understanding_executor
        self.data_discovery_executor = data_discovery_executor
        self.router = router
        self.csv_request_detector = csv_request_detector
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.llm_context_enhancer = llm_context_enhancer

    def configure_runtime(
        self,
        *,
        llm_service: Any,
        tool_registry: Any,
        llm_context_enhancer: Any = None,
    ) -> None:
        """고정 SQL 파이프라인이 사용할 Agent 소유 런타임을 연결한다."""
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.llm_context_enhancer = llm_context_enhancer

    @staticmethod
    def _sync_tool_context(
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> None:
        """run_sql/visualize_data가 선행 단계의 최신 근거를 읽도록 context를 동기화한다."""
        metadata = getattr(input.tool_context, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            input.tool_context.metadata = metadata
        metadata["main_workflow"] = state.to_metadata()
        metadata["structured_question"] = dict(state.structured_question)

    #     self._apply_subworkflow_limits(
    #         self.question_understanding_executor,
    #         max_steps=max_workflow_steps,
    #         retry_limit=workflow_retry_limit,
    #     )
    #     self._apply_subworkflow_limits(
    #         self.data_discovery_executor,
    #         max_steps=max_workflow_steps,
    #         retry_limit=workflow_retry_limit,
    #     )

    # @staticmethod
    # def _apply_subworkflow_limits(
    #     executor,
    #     *,
    #     max_steps: int,
    #     retry_limit: int,
    # ) -> None:
    #     if hasattr(executor, "max_steps"):
    #         executor.max_steps = max_steps
    #     if hasattr(executor, "retry_limit"):
    #         executor.retry_limit = retry_limit


    async def run(self, input: MainWorkflowInput) -> MainWorkflowTurnState:
        turn_history = input.metadata.get("turn_history", {})
        state = MainWorkflowTurnState(
            turn_id=input.request_id,
            original_question=input.original_message,
            conversation_id=input.conversation_id,
            turn_number=input.turn_number,
            history=(
                dict(turn_history)
                if isinstance(turn_history, dict)
                else {}
            ),
        )
        if self.csv_request_detector is not None:
            detected = self.csv_request_detector(input.original_message)
            state.csv_request = CsvRequestState(
                intent=detected.get("intent", "none"),
                filename=detected.get("filename"),
                resolution=detected.get("resolution"),
            )
            if (
                state.csv_request.intent == "use_as_input"
                and state.csv_request.filename is None
            ):
                history_csv_name = self._latest_history_csv_name(
                    state.history
                )
                if history_csv_name is not None:
                    state.csv_request.filename = history_csv_name
                    state.csv_request.resolution = (
                        "latest_successful_turn"
                    )
                else:
                    state.operation = "clarification_required"
                    state.result["message"] = (
                        "현재 대화에서 연속 분석에 사용할 수 있는 CSV 결과를 "
                        "찾지 못했습니다. 먼저 SQL 조회를 실행해 CSV 결과를 "
                        "생성하거나, 사용할 CSV 파일명을 알려주세요. "
                        "추가 도구를 호출하거나 SQL을 생성하지 말고 이 내용을 "
                        "사용자에게 안내하세요."
                    )
            if (
                state.csv_request.intent == "use_as_input"
                and state.csv_request.filename is not None
            ):
                state.operation = "continuous_analysis_dataset_required"
        _log_turn_state("initialized", state)

        if state.operation == "clarification_required":
            if self.llm_service is not None and self.tool_registry is not None:
                state.result["status"] = "failed"
                state.stage = "final"
            _log_turn_state("csv_request_unresolved", state)
            return state

        await self._run_question_understanding_with_fb1(input, state)
        _log_turn_state("question_understanding_saved", state)

        self._apply_required_dimension_clarification(state)

        is_non_sql_intent = self._is_non_sql_intent(state)
        if self._should_run_data_discovery(state):
            await self._run_data_discovery_with_fb2(input, state)
            _log_turn_state("data_discovery_saved", state)
        else:
            state.subflow("data_discovery").status = "skipped"
            if is_non_sql_intent:
                _log_turn_state_summary("data_discovery_skipped", state)
            else:
                _log_turn_state("data_discovery_saved", state)

        state.stage = "context_enrichment"
        if state.operation not in {
            "clarification_required",
            "continue_with_warning",
            "continuous_analysis_dataset_required",
        }:
            state.operation = "context_enrichment_ready"
        if is_non_sql_intent:
            _log_turn_state_summary("context_enrichment_ready", state)
        else:
            _log_turn_state("context_enrichment_ready", state)

        # 질문 이해와 데이터 탐색이 끝나면 Agent loop로 반환하지 않고, 런타임이
        # 연결된 경우 SQL 생성부터 시각화까지 정해진 후속 단계를 즉시 실행한다.
        if self.llm_service is not None and self.tool_registry is not None:
            await self._run_fixed_sql_pipeline(input, state)
        return state

    # sql 생성 흐름 고정 pipline, sql 질문이 아닌 경우 _is_non_sql_intent의 형식을 출력
    async def _run_fixed_sql_pipeline(
        self,
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> None:
        """자유형 Agent loop 없이 SQL 생성 → 검증 → 실행 → 시각화를 순서대로 수행한다."""
        if state.operation == "clarification_required":
            state.stage = "final"
            return

        if self._is_non_sql_intent(state):
            examples = self._general_examples(state.original_question)
            state.operation = "general_guidance"
            state.result["status"] = "success"
            state.result["message"] = "해당 Agent는 의료이용지표 Agent이기때문에 관련 질문을 해주세요."
            state.ui_components.append(UiComponent(
                rich_component=ButtonGroupComponent(
                    buttons=[
                        {"label": question, "action": question, "variant": "secondary"}
                        for question in examples
                    ],
                    orientation="vertical",
                    spacing="small",
                    alignment="stretch",
                    full_width=True,
                ),
                simple_component=None,
            ))
            state.stage = "final"
            return

        system_prompt = input.system_prompt or ""
        if self.llm_context_enhancer is not None:
            before = system_prompt
            try:
                system_prompt = await self.llm_context_enhancer.enhance_system_prompt_with_workflow(
                    system_prompt,
                    input.original_message,
                    input.tool_context.user,
                    workflow_state=state,
                    workflow_metadata=state.to_metadata(),
                )
                state.record_context_enrichment(
                    status="success",
                    enhancer=self.llm_context_enhancer.__class__.__name__,
                    system_prompt_before=before,
                    system_prompt_after=system_prompt,
                )
            except Exception as exc:
                state.record_context_enrichment(
                    status="failed",
                    enhancer=self.llm_context_enhancer.__class__.__name__,
                    system_prompt_before=before,
                    system_prompt_after=before,
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
                system_prompt = before
        else:
            state.record_context_enrichment(
                status="skipped",
                enhancer=None,
                system_prompt_before=system_prompt,
                system_prompt_after=system_prompt,
            )

        # 여러 도구 가운데 run_sql 하나만 SQL 생성 요청에 노출한다.
        run_sql_schema = next(
            (schema for schema in input.tool_schemas if schema.name == "run_sql"),
            None,
        )
        if run_sql_schema is None:
            self._fail_fixed_pipeline(state, "The run_sql tool is not available.")
            return

        # 이 지점부터 상태 전이는 LLM 선택이 아니라 아래 코드 순서에 의해 고정된다.
        state.stage = "sql_generation"
        state.operation = "sql_generation"
        sql_subflow = state.subflow("sql_processing")
        sql_subflow.status = "success"
        sql_subflow.current_node = "generate_validate_execute"
        sql_subflow.visited_nodes.append("generate_validate_execute")

        context_json = json.dumps(
            {
                "structured_question": state.structured_question,
                "metadata": state.metadata,
                "fewshot": state.fewshot,
            },
            ensure_ascii=False,
            default=str,
        )
        messages = [
            LlmMessage(
                role="user",
                content=(
                    f"{input.original_message}\n\nWorkflow context:\n{context_json}\n\n"
                    "Generate a read-only SQL query and return exactly one run_sql tool call."
                ),
            )
        ]

        last_error: str | None = None
        # 최초 실행과 1회 교정 재시도만 허용하여 무제한 tool-calling loop를 방지한다.
        for attempt_number in range(1, 3):
            request = LlmRequest(
                messages=messages,
                tools=[run_sql_schema],
                # 모델 응답을 run_sql 함수 호출로 제한한다.
                tool_choice={
                    "type": "function",
                    "function": {"name": "run_sql"},
                },
                user=input.tool_context.user,
                stream=False,
                temperature=0,
                system_prompt=system_prompt,
                metadata={"main_workflow": state.to_metadata()},
            )
            response = await self.llm_service.send_request(request)
            run_sql_calls = [
                call for call in (response.tool_calls or []) if call.name == "run_sql"
            ]
            # 호출이 없거나 여러 개이면 고정 파이프라인 계약 위반으로 처리한다.
            if len(run_sql_calls) != 1:
                received_names = [
                    call.name for call in (response.tool_calls or [])
                ]
                last_error = (
                    "SQL generation must return exactly one run_sql tool call. "
                    f"Received {len(run_sql_calls)} run_sql call(s); "
                    f"all tool calls={received_names}; "
                    f"finish_reason={response.finish_reason!r}."
                )
                state.record_sql_attempt(sql=None, status="failed", error_message=last_error)
            else:
                tool_call = run_sql_calls[0]
                sql = tool_call.arguments.get("sql") if isinstance(tool_call.arguments, dict) else None
                if not isinstance(sql, str) or not sql.strip():
                    last_error = "Generated run_sql call did not contain SQL."
                    state.record_sql_attempt(sql=None, status="failed", error_message=last_error)
                # 실행 전에 읽기 전용 SELECT인지 검사한다.
                elif get_statement_type(sql) != "SELECT":
                    last_error = "Only read-only SELECT statements are allowed."
                    state.record_sql_attempt(
                        sql=sql,
                        status="failed",
                        error_message=last_error,
                    )
                else:
                    # run_sql validates that metadata discovery succeeded in the
                    # current turn. Agent._send_message cannot publish this state
                    # after run() returns because the tool executes inside run().
                    self._sync_tool_context(input, state)
                    result = await self.tool_registry.execute(tool_call, input.tool_context)
                    state.record_sql_attempt(
                        sql=sql,
                        status="success" if result.success else "failed",
                        error_message=None if result.success else (result.error or result.result_for_llm),
                    )
                    if result.ui_component is not None:
                        state.ui_components.append(result.ui_component)
                    if result.success:
                        state.record_selected_metadata_from_sql(sql)
                        metadata = result.metadata or {}
                        state.result["csv_name"] = metadata.get("output_file")
                        state.result["row_count"] = metadata.get("row_count")
                        state.result["columns"] = list(metadata.get("columns") or [])
                        # SQL 성공 시 반환된 CSV를 입력으로 시각화를 즉시 연속 실행한다.
                        visualized = await self._visualize_sql_result(input, state)
                        sql_subflow.current_node = None
                        state.result["status"] = "success"
                        state.result["message"] = (
                            "SQL query and visualization completed successfully."
                            if visualized
                            else "SQL query completed successfully; no visualization was produced."
                        )
                        state.stage = "final"
                        state.operation = "final_response_ready"
                        return
                    last_error = result.error or result.result_for_llm

            if attempt_number == 1:
                messages.append(
                    LlmMessage(
                        role="user",
                        content=(
                            f"The previous SQL generation or execution failed: {last_error}. "
                            "Correct it and return exactly one run_sql tool call."
                        ),
                    )
                )

        sql_subflow.status = "failed"
        sql_subflow.current_node = None
        sql_subflow.errors.append(last_error or "SQL processing failed")
        self._fail_fixed_pipeline(state, last_error or "SQL processing failed")

    async def _visualize_sql_result(
        self,
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> bool:
        # 시각화도 별도 Agent 판단 없이 SQL 성공 직후 고정된 후속 단계로 실행한다.
        visualization = state.subflow("visualization")
        visualization.current_node = "visualize_data"
        visualization.visited_nodes.append("visualize_data")
        state.stage = "visualization"
        state.operation = "visualize_data"

        visualization_task = Task(
            title="시각화 실행",
            description="SQL 실행 결과를 표·차트·지도 대시보드로 변환합니다.",
            status="in_progress",
        )
        visualization_status = StatusCardComponent(
            title="시각화 실행",
            status="running",
            description="SQL 실행 결과를 시각화하고 있습니다.",
            icon="📊",
            metadata={"tool_name": "visualize_data"},
        )
        state.ui_components.extend(
            [
                UiComponent(
                    rich_component=TaskTrackerUpdateComponent.add_task(
                        visualization_task
                    )
                ),
                UiComponent(rich_component=visualization_status),
            ]
        )

        filename = state.result.get("csv_name")
        if not isinstance(filename, str) or not filename:
            visualization.status = "skipped"
            visualization.current_node = None
            state.ui_components.extend(
                [
                    UiComponent(
                        rich_component=TaskTrackerUpdateComponent.update_task(
                            visualization_task.id,
                            status="error",
                            detail="시각화할 SQL 결과 파일이 없습니다.",
                        )
                    ),
                    UiComponent(
                        rich_component=visualization_status.set_status(
                            "warning",
                            "시각화할 SQL 결과 파일이 없어 단계를 건너뛰었습니다.",
                        )
                    ),
                ]
            )
            return False

        result = await self.tool_registry.execute(
            ToolCall(
                id=f"visualize_{uuid.uuid4().hex[:8]}",
                name="visualize_data",
                arguments={"filename": filename, "title": input.original_message},
            ),
            input.tool_context,
        )
        visualization.status = "success" if result.success else "failed"
        visualization.current_node = None
        if not result.success:
            visualization.errors.append(result.error or result.result_for_llm)
        if result.ui_component is not None:
            state.ui_components.append(result.ui_component)
        completion_description = (
            "대화형 시각화 대시보드가 완료되었습니다."
            if result.success
            else f"시각화에 실패했습니다: {result.error or result.result_for_llm}"
        )
        state.ui_components.extend(
            [
                UiComponent(
                    rich_component=TaskTrackerUpdateComponent.update_task(
                        visualization_task.id,
                        status="completed" if result.success else "error",
                        detail=completion_description,
                    )
                ),
                UiComponent(
                    rich_component=visualization_status.set_status(
                        "success" if result.success else "error",
                        completion_description,
                    )
                ),
            ]
        )
        return result.success

    @staticmethod
    def _fail_fixed_pipeline(state: MainWorkflowTurnState, message: str) -> None:
        # 어느 고정 단계에서 실패해도 final로 수렴시켜 Agent loop 재진입을 막는다.
        state.result["status"] = "failed"
        state.result["message"] = message
        state.stage = "final"
        state.operation = "final_response_ready"

    @staticmethod
    def _latest_history_csv_name(
        history: dict[str, Any],
    ) -> str | None:
        """최신 구간부터 가장 최근의 성공 CSV 파일명을 찾습니다."""
        for section_name in ("latest", "recent", "older"):
            items = history.get(section_name, [])
            if not isinstance(items, list):
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue
                csv_name = item.get("csv_name")
                if isinstance(csv_name, str) and csv_name.strip():
                    return csv_name.strip()

        return None

    @staticmethod
    def _is_non_sql_intent(state: MainWorkflowTurnState) -> bool:
        question_subflow = state.subflow("question_understanding")
        intent = state.structured_question.get("intent")
        return (
            question_subflow.status == "success"
            and isinstance(intent, str)
            and intent != "sql"
        )

    async def _run_question_understanding_with_fb1(
        self,
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> dict[str, Any]:
        result_metadata = await self._run_question_understanding_once(input, state)
        failure_type = self._failure_type(result_metadata)

        if (
            result_metadata.get("status") == "failed"
            and failure_type in FB1_FAILURE_TYPES
            and state.fallback_state.can_use_fb1()
        ):
            feedback = self._build_fb1_feedback(result_metadata)
            state.fallback_state.mark_fb1(feedback)
            _log_turn_state("fb1_marked", state)
            result_metadata = await self._run_question_understanding_once(input, state)

        if (
            result_metadata.get("status") == "failed"
            and not self._is_json_validation_recovery_result(result_metadata)
        ):
            self._assign_clarification_result(
                state,
                failed_subflow="question_understanding",
                result_metadata=result_metadata,
            )

        return result_metadata

    async def _run_question_understanding_once(
        self,
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> dict[str, Any]:
        subflow_state = state.subflow("question_understanding")
        state.stage = "question_understanding"
        state.operation = "run_question_understanding"

        if self.question_understanding_executor is None:
            subflow_state.status = "skipped"
            return {"status": "skipped"}

        try:
            workflow_input = QuestUnderstand_Input(
                user_id=input.user_id,
                conversation_id=input.conversation_id,
                request_id=input.request_id,
                original_message=input.original_message,
                system_prompt=input.system_prompt,
                tool_names=[tool.name for tool in input.tool_schemas],
                metadata={
                    **input.metadata,
                    "fallback_state": state.fallback_state.snapshot(),
                    "fallback_feedback": state.fallback_state.active_feedback,
                },
            )
            result = await self.question_understanding_executor.run(workflow_input)
            self._collect_ui_component(state, result)
            result_metadata = self._to_metadata(result)

            structured_output = result_metadata.get("structured_output")
            state.structured_question = structured_output if isinstance(structured_output, dict) else {}
            state.metadata["searches"] = self._extract_searches(state.structured_question)
            _log_turn_state("structured_question_assigned", state)

            subflow_state.status = result_metadata.get("status", "success")
            subflow_state.errors = list(result_metadata.get("errors", []))
            subflow_state.retry_counts = dict(result_metadata.get("retry_counts", {}))
            subflow_state.failed_node_id = result_metadata.get("failed_node_id")
            subflow_state.failure_type = self._failure_type(result_metadata)
            subflow_state.failure_detail = result_metadata.get("failure_detail")
            return result_metadata
        except Exception as exc:
            subflow_state.status = "failed"
            subflow_state.errors.append(str(exc))
            subflow_state.failed_node_id = "question_understanding_subworkflow"
            subflow_state.failure_type = "question_structuring_failed"
            subflow_state.failure_detail = {
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            state.structured_question = {}
            result_metadata = {
                "status": "failed",
                "failed_node_id": "question_understanding_subworkflow",
                "failure_type": "question_structuring_failed",
                "errors": [f"question_understanding_failed: {type(exc).__name__}: {exc}"],
            }
            _log_turn_state("question_understanding_error_assigned", state)
            return result_metadata

    async def _run_data_discovery_with_fb2(
        self,
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> dict[str, Any]:
        result_metadata = await self._run_data_discovery_once(input, state)
        failure_type = self._failure_type(result_metadata)

        if result_metadata.get("status") != "failed":
            return result_metadata

        if failure_type in FB2_DATA_DISCOVERY_FAILURE_TYPES:
            if state.fallback_state.can_use_fb2():
                feedback = self._build_fb2_feedback(
                    result_metadata,
                    next_action="retry_data_discovery",
                    target_nodes=["metadata_search"],
                )
                state.fallback_state.mark_fb2(feedback)
                _log_turn_state("fb2_marked_retry_data_discovery", state)
                retry_result = await self._run_data_discovery_once(
                    input,
                    state,
                    structured_output_override=self._original_question_structured_output(state),
                    continue_with_warning=True,
                )
                if (
                    retry_result.get("status") == "failed"
                    and self._failure_type(retry_result) == "metadata_execution_error"
                ):
                    self._convert_metadata_execution_error_to_warning(state, retry_result)
                return retry_result

            self._convert_metadata_execution_error_to_warning(state, result_metadata)
            return result_metadata

        if failure_type in FB2_QUESTION_UNDERSTANDING_FAILURE_TYPES:
            if state.fallback_state.can_use_fb2():
                feedback = self._build_fb2_feedback(
                    result_metadata,
                    next_action="retry_question_understanding",
                    target_nodes=["question_structuring", "search_queries"],
                )
                state.fallback_state.mark_fb2(feedback)
                _log_turn_state("fb2_marked_retry_question_understanding", state)
                question_result = await self._run_question_understanding_once(input, state)
                if self._should_run_data_discovery(state):
                    retry_result = await self._run_data_discovery_once(input, state)
                    if retry_result.get("status") == "failed":
                        retry_failure_type = self._failure_type(retry_result)
                        if retry_failure_type == "metadata_execution_error":
                            self._convert_metadata_execution_error_to_warning(state, retry_result)
                        else:
                            self._assign_clarification_result(
                                state,
                                failed_subflow="data_discovery",
                                result_metadata=retry_result,
                            )
                    return retry_result
                self._assign_clarification_result(
                    state,
                    failed_subflow="question_understanding",
                    result_metadata=question_result,
                )
                return question_result

            self._assign_clarification_result(
                state,
                failed_subflow="data_discovery",
                result_metadata=result_metadata,
            )
            return result_metadata

        self._assign_clarification_result(
            state,
            failed_subflow="data_discovery",
            result_metadata=result_metadata,
        )
        return result_metadata

    async def _run_data_discovery_once(
        self,
        input: MainWorkflowInput,
        state: MainWorkflowTurnState,
        *,
        structured_output_override: dict[str, Any] | None = None,
        continue_with_warning: bool = False,
    ) -> dict[str, Any]:
        metadata_recovery = self._is_json_validation_recovery(state)
        if metadata_recovery:
            state.metadata["searches"] = []
            structured_output_override = self._json_validation_recovery_structured_output(state)

        subflow_state = state.subflow("data_discovery")
        state.stage = "data_discovery"
        state.operation = "run_data_discovery"

        if self.data_discovery_executor is None:
            subflow_state.status = "skipped"
            return {"status": "skipped"}

        try:
            workflow_input = self._build_data_discovery_input(
                state,
                structured_output_override=structured_output_override,
            )
            result = await self._run_data_discovery_executor(workflow_input, input, state)
            self._collect_ui_component(state, result)
            result_metadata = self._to_metadata(result)

            self._apply_data_discovery_result(state, result_metadata)
            if metadata_recovery:
                state.metadata["searches"] = []
            _log_turn_state("data_discovery_result_assigned", state)

            subflow_state.status = result_metadata.get("status", "success")
            if continue_with_warning and self._failure_type(result_metadata) == "metadata_execution_error":
                subflow_state.status = "success"
            subflow_state.errors = list(result_metadata.get("errors", []))
            subflow_state.retry_counts = dict(result_metadata.get("retry_counts", {}))
            subflow_state.failed_node_id = result_metadata.get("failed_node_id")
            subflow_state.failure_type = self._failure_type(result_metadata)
            subflow_state.failure_detail = result_metadata.get("failure_detail")
            return result_metadata
        except Exception as exc:
            subflow_state.status = "failed"
            subflow_state.errors.append(str(exc))
            subflow_state.failed_node_id = "data_discovery_subworkflow"
            subflow_state.failure_type = "metadata_execution_error"
            subflow_state.failure_detail = {
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            state.metadata = {"searches": [], "candidates": [], "selected": []}
            state.fewshot = []
            result_metadata = {
                "status": "failed",
                "failed_node_id": "data_discovery_subworkflow",
                "failure_type": "metadata_execution_error",
                "errors": [f"data_discovery_failed: {type(exc).__name__}: {exc}"],
            }
            _log_turn_state("data_discovery_error_assigned", state)
            return result_metadata

    def _build_data_discovery_input(
        self,
        state: MainWorkflowTurnState,
        *,
        structured_output_override: dict[str, Any] | None = None,
    ) -> Any:
        if DataDiscover_Input is None:
            return None

        question_subflow = state.subflow("question_understanding")
        structured_output = structured_output_override or state.structured_question
        metadata_recovery = (
            structured_output_override is not None
            and self._is_json_validation_recovery(state)
        )
        return DataDiscover_Input(
            status="success" if metadata_recovery else question_subflow.status,
            intent=structured_output.get("intent"),
            structured_output=structured_output,
            errors=question_subflow.errors,
            retry_counts=question_subflow.retry_counts,
        )

    @staticmethod
    def _is_json_validation_retry_exceeded(errors: list[str]) -> bool:
        error_text = "\n".join(str(error) for error in errors)
        return (
            "JSON_VALIDATION_FAILED: Generated JSON validation failed after retry limit."
            in error_text
        )

    def _is_json_validation_recovery_result(self, result_metadata: dict[str, Any]) -> bool:
        return (
            self._failure_type(result_metadata) == "json_validation_retry_exceeded"
            or self._is_json_validation_retry_exceeded(
                list(result_metadata.get("errors", []))
            )
        )

    def _is_json_validation_recovery(self, state: MainWorkflowTurnState) -> bool:
        question_subflow = state.subflow("question_understanding")
        return (
            question_subflow.status == "failed"
            and (
                question_subflow.failure_type == "json_validation_retry_exceeded"
                or self._is_json_validation_retry_exceeded(question_subflow.errors)
            )
            and bool(state.original_question.strip())
        )

    async def _run_data_discovery_executor(
        self,
        workflow_input: Any,
        main_input: MainWorkflowInput,
        state: MainWorkflowTurnState,
    ) -> Any:
        if workflow_input is not None:
            try:
                return await self.data_discovery_executor.run(workflow_input)
            except TypeError:
                logger.debug(
                    "data_discovery_executor rejected DataDiscover_Input; retrying with MainWorkflowInput and TurnState",
                    exc_info=True,
                )
        return await self.data_discovery_executor.run(main_input, state)

    def _should_run_data_discovery(self, state: MainWorkflowTurnState) -> bool:
        if state.operation == "clarification_required":
            return False
        question_subflow = state.subflow("question_understanding")
        if (
            question_subflow.status == "success"
            and state.structured_question.get("intent") == "sql"
            and bool(state.structured_question)
        ):
            return True

        return (
            question_subflow.status == "failed"
            and (
                question_subflow.failure_type == "json_validation_retry_exceeded"
                or self._is_json_validation_retry_exceeded(question_subflow.errors)
            )
            and bool(state.original_question.strip())
        )

    # sql(통계지표)와 관련없는 질문을 헀을 때, 제시하는 기본 3가지 예시 가이드 질문
    @staticmethod
    def _general_examples(question: str) -> list[str]:
        """관련 의료 키워드가 있으면 같은 주제의 예시를, 없으면 기본 예시를 반환합니다."""
        normalized = str(question or "").casefold()
        topic_examples = (
            (("항생제", "antibiotic"), [
                "2024년 전국 항생제 처방률을 보여줘.",
                "2024년 시도별 항생제 처방률을 보여줘.",
                "연도별 전국 항생제 처방률 추이를 보여줘.",
            ]),
            (("진료비", "의료비"), [
                "2024년 전국 총 진료비를 보여줘.",
                "2024년 시도별 총 진료비를 보여줘.",
                "전국 연도별 총 진료비 추이를 보여줘.",
            ]),
            (("검진", "건강검진"), [
                "2024년 전국 건강검진 수검자 수를 보여줘.",
                "2024년 시도별 건강검진 수검자 수를 보여줘.",
                "연도별 건강검진 수검자 수를 보여줘.",
            ]),
            (("요양기관", "병원", "의료기관"), [
                "2024년 전국 요양기관 수를 보여줘.",
                "2024년 시도별 요양기관 수를 보여줘.",
                "전국 연도별 요양기관 수 추이를 보여줘.",
            ]),
        )
        for keywords, examples in topic_examples:
            if any(keyword in normalized for keyword in keywords):
                return examples
        return [
            "전국 요양기관 수를 보여줘.",
            "연도별 건강검진 수검자 수를 보여줘.",
            "시도별 의료이용 현황을 보여줘.",
        ]

    @staticmethod
    def _apply_required_dimension_clarification(state: MainWorkflowTurnState) -> None:
        """통계 질문의 연도와 지역 범위를 SQL 생성 전에 확정합니다."""
        if state.subflow("question_understanding").status != "success":
            return
        structured = state.structured_question
        if structured.get("intent") != "sql":
            return

        question = state.original_question.strip()
        group_by = structured.get("group_by") if isinstance(structured.get("group_by"), list) else []
        time_conditions = structured.get("time_conditions") if isinstance(structured.get("time_conditions"), list) else []
        group_text = " ".join(
            " ".join(str(item.get(key) or "") for key in ("field_meaning", "original_text"))
            for item in group_by if isinstance(item, dict)
        )
        requests_time_series = any(term in question or term in group_text for term in ("연도별", "년도별", "해마다", "매년"))
        has_year = bool(time_conditions) or bool(re.search(r"(?:19|20)\d{2}\s*년?", question)) or requests_time_series
        region_names = (
            "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
            "경기", "강원", "충북", "충청북도", "충남", "충청남도", "전북",
            "전라북도", "전남", "전라남도", "경북", "경상북도", "경남",
            "경상남도", "제주",
        )
        has_region = (
            "전국" in question
            or any(region in question for region in region_names)
            or any(term in question for term in ("시도별", "시·도별", "지역별", "시군구별", "읍면동별", "중진료권별"))
            or any(term in group_text for term in ("지역", "시도", "시·도", "시군구", "읍면동", "중진료권"))
        )

        if not has_year:
            state.operation = "clarification_required"
            state.result["status"] = "success"
            state.result["message"] = "조회할 연도를 지정해 주세요. 예: 2024년 전국 총 진료비를 보여줘."
            state.stage = "final"
            return
        if not has_region:
            state.operation = "clarification_required"
            state.result["status"] = "success"
            state.result["message"] = "조회할 지역 범위를 지정해 주세요. 전국인지, 특정 시·도 또는 시군구인지 알려주세요."
            state.stage = "final"

    def _apply_data_discovery_result(
        self,
        state: MainWorkflowTurnState,
        result_metadata: dict[str, Any],
    ) -> None:
        metadata_output = result_metadata.get("metadata_output")
        fewshot_output = result_metadata.get("fewshot_output")

        state.metadata = {
            "searches": list(
                state.metadata.get("searches")
                or self._extract_searches(state.structured_question)
            ),
            "candidates": self._extract_metadata_candidates(metadata_output),
            "selected": list(state.metadata.get("selected", [])),
        }
        state.fewshot = self._extract_fewshot_examples(fewshot_output)

    def _original_question_structured_output(self, state: MainWorkflowTurnState) -> dict[str, Any]:
        return {
            "intent": "sql",
            "question": state.original_question,
            "original_question": state.original_question,
        }

    def _json_validation_recovery_structured_output(
        self,
        state: MainWorkflowTurnState,
    ) -> dict[str, Any]:
        structured_output = {
            key: value
            for key, value in state.structured_question.items()
            if key not in {"search_plan", "search_queries"}
        }
        question_subflow = state.subflow("question_understanding")
        error_message = self._json_validation_recovery_error_message(question_subflow)
        structured_output.update(
            {
                "intent": "sql",
                "question": state.original_question,
                "original_question": state.original_question,
                "metadata_discovery_recovery": {
                    "source_subflow": "question_understanding",
                    "source_node_id": question_subflow.failed_node_id,
                    "failure_type": question_subflow.failure_type,
                    "error_message": error_message,
                },
            }
        )
        if error_message:
            structured_output["metadata_discovery_question"] = (
                f"{state.original_question}\n\n"
                f"JSON validation error: {error_message}"
            )
        return structured_output

    @staticmethod
    def _json_validation_recovery_error_message(subflow_state: Any) -> str | None:
        failure_detail = getattr(subflow_state, "failure_detail", None)
        if isinstance(failure_detail, dict):
            error = failure_detail.get("error")
            if isinstance(error, str) and error.strip():
                return error.strip()
        for error in getattr(subflow_state, "errors", []) or []:
            if isinstance(error, str) and error.strip():
                return error.strip()
        return None

    def _convert_metadata_execution_error_to_warning(
        self,
        state: MainWorkflowTurnState,
        result_metadata: dict[str, Any],
    ) -> None:
        errors = list(result_metadata.get("errors", []))
        warning = (
            "\n".join(str(error) for error in errors)
            if errors
            else "metadata_execution_error"
        )
        state.result["message"] = (
            f"{warning}\n"
            "위 에러로 인해 metadata 검색을 완료하지 못했습니다.\n"
            "[필수 규칙] run_sql을 호출하기 전에 반드시 search_business_metadata tool을 먼저 호출해 "
            "전체 table metadata와 column metadata를 다시 검색하세요. "
            "이 규칙을 만족하기 전에는 SQL 생성/검증/실행을 진행하면 안 됩니다."
        )
        state.operation = "continue_with_warning"
        state.subflow("data_discovery").status = "success"

    @staticmethod
    def _collect_ui_component(state: MainWorkflowTurnState, result: Any) -> None:
        ui_component = getattr(result, "ui_component", None)
        if ui_component is not None:
            state.ui_components.append(ui_component)

    def _assign_clarification_result(
        self,
        state: MainWorkflowTurnState,
        *,
        failed_subflow: str,
        result_metadata: dict[str, Any],
    ) -> None:
        failure_type = self._failure_type(result_metadata) or "unknown_failure"
        failed_node_id = result_metadata.get("failed_node_id") or failed_subflow
        errors = result_metadata.get("errors") or []
        error_text = "\n".join(str(error) for error in errors) or failure_type
        state.operation = "clarification_required"
        state.result["message"] = (
            f"{failed_subflow}의 {failed_node_id}에서 {failure_type} 오류가 발생했습니다.\n"
            f"원인: {error_text}\n"
            "사용자에게 어떤 조건, 지표, 테이블/컬럼 또는 시간 범위가 필요한지 "
            "명확히 재질문하세요."
        )

    @staticmethod
    def _build_fb1_feedback(result_metadata: dict[str, Any]) -> dict[str, Any]:
        failure_type = MainWorkflowExecutor._failure_type(result_metadata) or "question_understanding_failed"
        return {
            "fallback_id": "FB1",
            "failed_subflow": "question_understanding",
            "failed_node_id": result_metadata.get("failed_node_id"),
            "reason": failure_type,
            "next_action": "retry_question_understanding",
            "feedback": {
                "reason": failure_type,
                "errors": list(result_metadata.get("errors", [])),
                "failure_detail": result_metadata.get("failure_detail"),
            },
        }

    @staticmethod
    def _build_fb2_feedback(
        result_metadata: dict[str, Any],
        *,
        next_action: str,
        target_nodes: list[str],
    ) -> dict[str, Any]:
        failure_type = MainWorkflowExecutor._failure_type(result_metadata) or "data_discovery_failed"
        return {
            "fallback_id": "FB2",
            "failed_subflow": "data_discovery",
            "failed_node_id": result_metadata.get("failed_node_id"),
            "reason": failure_type,
            "next_action": next_action,
            "feedback": {
                "reason": failure_type,
                "target_nodes": target_nodes,
                "errors": list(result_metadata.get("errors", [])),
                "failure_detail": result_metadata.get("failure_detail"),
            },
        }

    @staticmethod
    def _failure_type(result_metadata: dict[str, Any]) -> str | None:
        failure_type = result_metadata.get("failure_type")
        if isinstance(failure_type, str) and failure_type:
            return failure_type
        joined_errors = "\n".join(str(error) for error in result_metadata.get("errors", []))
        for candidate in (
            "metadata_semantic_mismatch",
            "metadata_execution_error",
            "intent_classification_failed",
            "time_normalization_failed",
            "time_metadata_validation_failed",
            "question_structuring_failed",
            "search_queries_generation_failed",
            "json_validation_failed",
            "json_validation_retry_exceeded",
        ):
            if candidate in joined_errors:
                return candidate
        return None

    @staticmethod
    def _extract_searches(structured_question: dict[str, Any]) -> list[dict[str, Any]]:
        search_plan = structured_question.get("search_plan")
        if isinstance(search_plan, dict):
            for key in ("queries", "search_queries"):
                queries = search_plan.get(key)
                if isinstance(queries, list):
                    return [query for query in queries if isinstance(query, dict)]

        queries = structured_question.get("search_queries")
        if isinstance(queries, list):
            return [query for query in queries if isinstance(query, dict)]
        return []

    @staticmethod
    def _extract_metadata_candidates(metadata_output: Any) -> list[dict[str, Any]]:
        if not isinstance(metadata_output, dict):
            return []

        candidates: list[dict[str, Any]] = []
        for table in metadata_output.get("tables", []) or []:
            if isinstance(table, dict):
                candidates.append({"type": "table", **table})
        for column in metadata_output.get("columns", []) or []:
            if isinstance(column, dict):
                candidates.append({"type": "column", **column})
        return candidates

    @staticmethod
    def _extract_fewshot_examples(fewshot_output: Any) -> list[dict[str, Any]]:
        if not isinstance(fewshot_output, dict):
            return []
        examples = fewshot_output.get("examples")
        if not isinstance(examples, list):
            return []
        return [example for example in examples if isinstance(example, dict)]

    @staticmethod
    def _to_metadata(result: Any) -> dict[str, Any]:
        if hasattr(result, "to_metadata"):
            return result.to_metadata()
        if isinstance(result, dict):
            return result
        return {"status": getattr(result, "status", "success")}



