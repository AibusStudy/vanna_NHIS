import pandas as pd

from vanna.integrations.chartjs import VisualizationDashboardPlanner


def test_multiple_series_does_not_create_misleading_composition() -> None:
    dashboard = VisualizationDashboardPlanner().plan(
        pd.DataFrame({"year": [2023, 2024], "admitted": [10, 12], "discharged": [8, 9]}),
        title="입원 환자 추이",
        column_metadata={},
        structured_question={},
    )

    assert "composition-chart" not in {view["id"] for view in dashboard["views"]}


def test_map_requires_verified_geographic_field() -> None:
    dashboard = VisualizationDashboardPlanner().plan(
        pd.DataFrame({"category": ["A"], "count": [10]}),
        title="지역별 환자 수",
        column_metadata={},
        structured_question={"requested_views": ["table", "map"], "geo_level": "시도"},
    )

    assert not any(view["type"] == "map" for view in dashboard["views"])
    assert dashboard["warnings"]


def test_verified_region_field_adds_map_contract() -> None:
    dashboard = VisualizationDashboardPlanner().plan(
        pd.DataFrame({"sido_cd": ["11", "26"], "count": [10, 20]}),
        title="지역별 환자 수",
        column_metadata={"sido_cd": {"logical_name": "시도 코드", "semantic_type": "region"}},
        structured_question={"requested_views": ["table", "map"], "geo_level": "시도"},
    )

    map_view = next(view for view in dashboard["views"] if view["type"] == "map")
    assert map_view["geo"]["regionCodeField"] == "sido_cd"
