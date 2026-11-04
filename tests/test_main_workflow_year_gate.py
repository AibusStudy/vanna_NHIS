from types import SimpleNamespace

from vanna.core.main_workflow.excutor import MainWorkflowExecutor


def _state(question: str, *, group_by: list[dict] | None = None):
    question_understanding = SimpleNamespace(status="success")
    return SimpleNamespace(
        subflow=lambda _: question_understanding,
        structured_question={
            "intent": "sql",
            "time_conditions": [],
            "group_by": group_by or [],
        },
        original_question=question,
        operation="context_enrichment_ready",
        result={},
        stage="context_enrichment",
    )


def test_missing_year_does_not_require_clarification() -> None:
    state = _state("서울과 부산의 인구 천 명당 허가 병상 수를 비교해줘.")

    MainWorkflowExecutor._apply_required_dimension_clarification(state)

    assert state.operation == "context_enrichment_ready"
    assert state.result == {}
    assert state.stage == "context_enrichment"


def test_missing_region_still_requires_clarification() -> None:
    state = _state("인구 천 명당 허가 병상 수를 보여줘.")

    MainWorkflowExecutor._apply_required_dimension_clarification(state)

    assert state.operation == "clarification_required"
    assert "지역 범위" in state.result["message"]
    assert state.stage == "final"
