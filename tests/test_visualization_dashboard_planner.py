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
    assert [view["type"] for view in dashboard["views"]] == ["table", "chartjs", "summary"]


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
    assert map_view["geo"]["valueField"] == "count"


def test_coordinates_add_executable_heatmap_contract() -> None:
    dashboard = VisualizationDashboardPlanner().plan(
        pd.DataFrame({"latitude": [37.56], "longitude": [126.97], "count": [25]}),
        title="환자 밀집도",
        column_metadata={
            "latitude": {"semantic_type": "latitude"},
            "longitude": {"semantic_type": "longitude"},
            "count": {"logical_name": "환자 수", "unit": "명"},
        },
        structured_question={"requested_views": ["heatmap"]},
    )

    heatmap = next(view for view in dashboard["views"] if view["type"] == "heatmap")
    assert heatmap["geo"]["latitudeField"] == "latitude"
    assert heatmap["geo"]["longitudeField"] == "longitude"
    assert heatmap["geo"]["valueField"] == "count"
    assert heatmap["geo"]["valueLabel"] == "환자 수"


def test_nationwide_result_creates_exact_three_view_dashboard() -> None:
    dashboard = VisualizationDashboardPlanner().plan(
        pd.DataFrame({
            "STD_YYYY": ["2023", "2023", "2023"],
            "ZN_TYPE_CD": ["0", "1", "1"],
            "SI_DO_CD": [None, "11", "26"],
            "IDCT_VL": [100, 60, 40],
        }),
        title="요양기관 현황",
        column_metadata={
            "STD_YYYY": {"logical_name": "기준연도", "sql_role": "time", "semantic_type": "year"},
            "ZN_TYPE_CD": {"logical_name": "지역구분코드", "sql_role": "code", "semantic_type": "code"},
            "SI_DO_CD": {"logical_name": "시도", "sql_role": "dimension", "semantic_type": "administrative_area"},
            "IDCT_VL": {"logical_name": "요양기관 수", "sql_role": "measure", "semantic_type": "quantity", "unit": "개"},
        },
        structured_question={
            "geo_level": "시도",
            "requested_views": ["table", "bar", "map"],
            "visualization_plan": {"include_total": True},
        },
    )

    assert [view["type"] for view in dashboard["views"]] == ["table", "chartjs", "map"]
    assert dashboard["views"][1]["chartType"] == "bar"
    assert dashboard["views"][2]["geo"]["regionLevelField"] == "ZN_TYPE_CD"
    assert dashboard["views"][2]["geo"]["yearField"] == "STD_YYYY"
    assert dashboard["layout"]["type"] == "split-3"


def test_numeric_std_yyyy_is_not_selected_as_map_metric() -> None:
    dashboard = VisualizationDashboardPlanner().plan(
        pd.DataFrame({
            "STD_YYYY": [2021, 2021],
            "SI_DO_CD": ["11", "26"],
            "IDCT_VL": [12.5, 9.8],
        }),
        title="항생제 처방률",
        column_metadata={
            "STD_YYYY": {"logical_name": "기준연도", "sql_role": "time", "semantic_type": "year"},
            "SI_DO_CD": {"logical_name": "시도", "sql_role": "dimension", "semantic_type": "administrative_area"},
            "IDCT_VL": {"logical_name": "항생제 처방률", "sql_role": "measure", "semantic_type": "ratio", "unit": "%"},
        },
        structured_question={"requested_views": ["table", "map"], "geo_level": "시도"},
    )

    map_view = next(view for view in dashboard["views"] if view["type"] == "map")
    assert map_view["geo"]["yearField"] == "STD_YYYY"
    assert map_view["geo"]["valueField"] == "IDCT_VL"
