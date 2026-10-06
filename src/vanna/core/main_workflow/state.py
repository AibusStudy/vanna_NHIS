from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vanna.core.tool import ToolContext, ToolSchema

MainWorkflowStatus = Literal["success", "failed", "skipped"]
MainWorkflowStage = Literal[
    "question_understanding",
    "data_discovery",
    "context_enrichment",
    "sql_generation",
    "sql_regeneration",
    "successful_query_save",
    "visualization",
    "final_answer",
    "final",
]
SqlAttemptStatus = Literal["success", "failed"]
CsvRequestIntent = Literal[
    "none",
    "use_as_input",
    "save_as_output",
    "ambiguous",
]


@dataclass
class FallbackState:
    fb1_count: int = 0
    fb2_count: int = 0
    active_feedback: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fb1_used(self) -> bool:
        return self.fb1_count >= 1

    @property
    def fb2_used(self) -> bool:
        return self.fb2_count >= 1

    def can_use_fb1(self) -> bool:
        return self.fb1_count < 1

    def can_use_fb2(self) -> bool:
        return self.fb2_count < 1

    def mark_fb1(self, feedback: dict[str, Any]) -> None:
        self.fb1_count += 1
        self.active_feedback = feedback
        self.history.append(feedback)

    def mark_fb2(self, feedback: dict[str, Any]) -> None:
        self.fb2_count += 1
        self.active_feedback = feedback
        self.history.append(feedback)

    def snapshot(self) -> dict[str, Any]:
        return {
            "fb1_used": self.fb1_used,
            "fb2_used": self.fb2_used,
            "fb1_count": self.fb1_count,
            "fb2_count": self.fb2_count,
            "active_feedback": self.active_feedback,
        }


@dataclass
class SubworkflowState:
    status: MainWorkflowStatus = "skipped"
    current_node: str | None = None
    visited_nodes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    retry_counts: dict[str, int] = field(default_factory=dict)
    failed_node_id: str | None = None
    failure_type: str | None = None
    failure_detail: dict[str, Any] | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "current_node": self.current_node,
            "visited_nodes": list(self.visited_nodes),
            "errors": list(self.errors),
            "retry_counts": dict(self.retry_counts),
            "failed_node_id": self.failed_node_id,
            "failure_type": self.failure_type,
            "failure_detail": self.failure_detail,
        }


@dataclass
class SqlAttemptState:
    attempt_number: int
    sql: str | None = None
    error_message: str | None = None
    status: SqlAttemptStatus = "failed"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "sql": self.sql,
            "error_message": self.error_message,
            "status": self.status,
        }


@dataclass
class CsvRequestState:
    """현재 사용자 질문에서 판별한 CSV 요청의 종류와 파일 정보입니다."""

    intent: CsvRequestIntent = "none"
    filename: str | None = None
    resolution: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "filename": self.filename,
            "resolution": self.resolution,
        }


@dataclass
class ActiveDatasetState:
    """CsvToJsonTool이 실제로 준비한 Dataset의 작은 참조 정보입니다."""

    dataset_id: str
    source_csv: str
    schema_file: str
    schema: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_csv": self.source_csv,
            "schema_file": self.schema_file,
            "schema": dict(self.schema),
        }


@dataclass(frozen=True)
class MainWorkflowInput:
    user_id: str
    conversation_id: str
    request_id: str
    original_message: str
    system_prompt: str | None
    tool_schemas: list[ToolSchema]
    tool_context: ToolContext
    turn_number: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MainWorkflowTurnState:
    turn_id: str
    original_question: str
    conversation_id: str | None = None
    turn_number: int = 1
    history: dict[str, Any] = field(default_factory=dict)
    csv_request: CsvRequestState = field(default_factory=CsvRequestState)
    active_dataset: ActiveDatasetState | None = None

    stage: MainWorkflowStage = "question_understanding"
    operation: str | None = None

    fallback_state: FallbackState = field(default_factory=FallbackState)

    subflows: dict[str, SubworkflowState] = field(
        default_factory=lambda: {
            "question_understanding": SubworkflowState(),
            "data_discovery": SubworkflowState(),
            "sql_processing": SubworkflowState(),
        }
    )

    structured_question: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(
        default_factory=lambda: {
            "searches": [],
            "candidates": [],
            "selected": [],
        }
    )

    fewshot: list[dict[str, Any]] = field(default_factory=list)

    ui_components: list[Any] = field(default_factory=list)

    context_enrichment: dict[str, Any] = field(
        default_factory=lambda: {
            "status": "skipped",
            "enhancer": None,
            "input_summary": {},
            "output_summary": {},
            "warnings": [],
            "errors": [],
        }
    )

    current_attempt_number: int = 0
    attempts: list[SqlAttemptState] = field(default_factory=list)

    # final_answer까지 도달하지 못했을 때 사용할 사용자 안내 후보입니다.
    # Context에는 주입하지 않고 턴 종료 시 result.message가 비어 있을 때만 사용합니다.
    pending_message: str | None = None

    result: dict[str, Any] = field(
        default_factory=lambda: {
            "message": None,
            "csv_name": None,
            "json_name": None,
            "row_count": None,
            "columns": [],
            "status": None,
        }
    )

    def subflow(self, name: str) -> SubworkflowState:
        if name not in self.subflows:
            raise ValueError(f"Unsupported subflow: {name}")
        return self.subflows[name]

    def subworkflow(self, name: str) -> SubworkflowState:
        return self.subflow(name)

    def record_context_enrichment(
        self,
        *,
        status: str,
        enhancer: str | None,
        system_prompt_before: str | None,
        system_prompt_after: str | None,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.context_enrichment = {
            "status": status,
            "enhancer": enhancer,
            "input_summary": {
                "has_structured_question": bool(self.structured_question),
                "metadata_candidate_count": len(self.metadata.get("candidates", [])),
                "fewshot_count": len(self.fewshot),
                "system_prompt_length_before": len(system_prompt_before or ""),
            },
            "output_summary": {
                "system_prompt_length_after": len(system_prompt_after or ""),
                "changed": (system_prompt_before or "") != (system_prompt_after or ""),
            },
            "warnings": list(warnings or []),
            "errors": list(errors or []),
        }

    def record_selected_metadata_from_sql(
        self, sql: str | None
    ) -> list[dict[str, Any]]:
        if not sql:
            return []

        sql_upper = str(sql).upper()
        candidates = self.metadata.get("candidates", [])
        if not isinstance(candidates, list):
            return []

        selected: list[dict[str, Any]] = []
        seen: set[tuple[str, str | None, str | None]] = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            candidate_type = str(candidate.get("type") or "").lower()
            table_name = candidate.get("table_name")
            column_name = candidate.get("column_name")

            table_text = str(table_name).strip().upper() if table_name else ""
            column_text = str(column_name).strip().upper() if column_name else ""

            matched = False
            if candidate_type == "table" and table_text:
                matched = table_text in sql_upper
            elif candidate_type == "column" and column_text:
                matched = column_text in sql_upper
            elif table_text or column_text:
                matched = bool(table_text and table_text in sql_upper) or bool(
                    column_text and column_text in sql_upper
                )

            if not matched:
                continue

            key = (candidate_type, table_text or None, column_text or None)
            if key in seen:
                continue
            seen.add(key)

            selected_item = dict(candidate)
            selected_item["selection_source"] = "run_sql"
            selected.append(selected_item)

        self.metadata["selected"] = selected
        return selected

    def record_sql_attempt(
        self,
        *,
        sql: str | None,
        status: SqlAttemptStatus,
        error_message: str | None = None,
    ) -> SqlAttemptState:
        self.current_attempt_number += 1
        attempt = SqlAttemptState(
            attempt_number=self.current_attempt_number,
            sql=sql,
            error_message=error_message,
            status=status,
        )
        self.attempts.append(attempt)
        # result에는 가장 최근 SQL 실행 시도의 성공/실패 상태를 유지합니다.
        self.result["status"] = attempt.status
        return attempt

    def record_active_dataset(
        self,
        *,
        dataset_id: str,
        source_csv: str,
        schema_file: str,
        schema: dict[str, Any],
    ) -> ActiveDatasetState:
        """CsvToJsonTool이 준비한 Dataset의 참조 정보만 기록합니다."""
        active_dataset = ActiveDatasetState(
            dataset_id=dataset_id,
            source_csv=source_csv,
            schema_file=schema_file,
            schema=dict(schema),
        )
        self.active_dataset = active_dataset

        # 사용자가 "이 CSV"라고만 말해 파일명이 미해결이었던 경우,
        # 실제로 준비된 파일명으로 판별 상태도 함께 확정합니다.
        if self.csv_request.filename is None:
            self.csv_request.filename = source_csv
            self.csv_request.resolution = "tool_resolved_filename"

        return active_dataset

    def to_metadata(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "turn_number": self.turn_number,
            "history": dict(self.history),
            "csv_request": self.csv_request.to_metadata(),
            "active_dataset": (
                self.active_dataset.to_metadata()
                if self.active_dataset is not None
                else None
            ),
            "original_question": self.original_question,
            "stage": self.stage,
            "operation": self.operation,
            "fallback_state": {
                **self.fallback_state.snapshot(),
                "history": list(self.fallback_state.history),
            },
            "subflows": {
                name: subflow.to_metadata() for name, subflow in self.subflows.items()
            },
            "structured_question": dict(self.structured_question),
            "metadata": {
                "searches": list(self.metadata.get("searches", [])),
                "candidates": list(self.metadata.get("candidates", [])),
                "selected": list(self.metadata.get("selected", [])),
            },
            "fewshot": list(self.fewshot),
            "context_enrichment": dict(self.context_enrichment),
            "current_attempt_number": self.current_attempt_number,
            "attempts": [attempt.to_metadata() for attempt in self.attempts],
            "pending_message": self.pending_message,
            "result": dict(self.result),
        }

    ## 메타데이터 검색 결과 병합
    def add_metadata_search_result(
        self,
        *,
        searches: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> None:
        existing_searches = self.metadata.setdefault(
            "searches",
            [],
        )
        existing_candidates = self.metadata.setdefault(
            "candidates",
            [],
        )

        existing_search_keys = {
            (
                item.get("query"),
                item.get("scope"),
            )
            for item in existing_searches
            if isinstance(item, dict)
        }

        for search in searches:
            if not isinstance(search, dict):
                continue

            search_key = (
                search.get("query"),
                search.get("scope"),
            )

            if search_key in existing_search_keys:
                continue

            existing_searches.append(dict(search))
            existing_search_keys.add(search_key)

        candidate_index = {
            self._metadata_candidate_key(candidate): index
            for index, candidate in enumerate(existing_candidates)
            if isinstance(candidate, dict)
        }

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            candidate_key = self._metadata_candidate_key(candidate)

            if candidate_key is None:
                continue

            existing_index = candidate_index.get(candidate_key)

            if existing_index is None:
                existing_candidates.append(dict(candidate))
                candidate_index[candidate_key] = len(existing_candidates) - 1
                continue

            existing = existing_candidates[existing_index]

            # 새 검색에서 추가된 정보가 있으면 기존 후보를 보강
            existing.update(
                {key: value for key, value in candidate.items() if value is not None}
            )

    ## 메타데이터 중복 판별 키
    @staticmethod
    def _metadata_candidate_key(
        candidate: dict[str, Any],
    ) -> tuple[Any, ...] | None:
        candidate_type = candidate.get("type")

        if candidate_type == "table":
            table_name = candidate.get("table_name")
            if not table_name:
                return None

            return (
                "table",
                table_name,
            )

        if candidate_type == "column":
            table_name = candidate.get("table_name")
            column_name = candidate.get("column_name")

            if not table_name or not column_name:
                return None

            return (
                "column",
                table_name,
                column_name,
            )

        return None

    ## few shot 저장
    def add_fewshot_results(
        self,
        examples: list[dict[str, Any]],
    ) -> None:
        existing_keys = {
            self._fewshot_key(example)
            for example in self.fewshot
            if isinstance(example, dict)
        }

        for example in examples:
            if not isinstance(example, dict):
                continue

            example_key = self._fewshot_key(example)

            if example_key is None:
                continue

            if example_key in existing_keys:
                continue

            self.fewshot.append(dict(example))
            existing_keys.add(example_key)

    ## few shot 중복 식별키
    # 현재 질문과 sql 쌍으로 존재
    # 향후 번호 등을 활용해서 더 간단한 방법으로 중복 거르는 방법에 대해 찾아야함
    @staticmethod
    def _fewshot_key(
        example: dict[str, Any],
    ) -> tuple[str, str] | None:
        question = example.get("question")
        args = example.get("args")

        if not isinstance(question, str):
            return None

        if not isinstance(args, dict):
            return None

        sql = args.get("sql")

        if not isinstance(sql, str):
            return None

        return (
            question.strip(),
            sql.strip(),
        )
