"""Responsive Chart.js configuration generator for SQL result data."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


class ChartJsChartGenerator:
    """Build JSON-serializable Chart.js data and options from a DataFrame."""

    COLOR_PALETTE = ["#B5121B", "#CF2E2E", "#E34A3F", "#E96B57", "#F08D75", "#F3B09E"]
    TEXT_COLOR = "#242424"
    MUTED_TEXT_COLOR = "#6B6B6B"

    def generate_chart(
        self,
        df: pd.DataFrame,
        title: str = "데이터 시각화",
        column_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if df.empty:
            raise ValueError("Cannot visualize empty DataFrame")
        self.column_metadata = column_metadata or {}
        chart_type, data, axis_titles, values = self._create_chart(df.copy())
        options = self._options(title, chart_type, axis_titles, values)
        return {"chart_type": chart_type, "data": data, "options": options}

    def _create_chart(
        self, df: pd.DataFrame
    ) -> tuple[str, dict[str, Any], tuple[str, str], list[float]]:
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_dimensions = [column for column in numeric if self._is_dimension(column)]
        measures = [column for column in numeric if column not in numeric_dimensions]
        dimensions = categorical + numeric_dimensions

        if len(df) == 1 and measures:
            labels = [self._label(column) for column in measures[:6]]
            values = [self._number(df.iloc[0][column]) for column in measures[:6]]
            details = [self._detail(column, label) for column, label in zip(measures[:6], labels)]
            return "bar", {
                "labels": labels,
                "datasets": [self._dataset("결과값", values, 0, details)],
            }, ("항목", "값"), values

        if len(measures) == 1 and len(dimensions) >= 2:
            value = measures[0]
            time_dimensions = [column for column in dimensions if self._is_time_dimension(column)]
            series = time_dimensions[0] if time_dimensions else dimensions[1]
            category = next(column for column in dimensions if column != series)
            category_values = list(dict.fromkeys(df[category].tolist()))
            labels = [self._display_value(category, item) for item in category_values]
            datasets = []
            all_values: list[float] = []
            for index, series_value in enumerate(sorted(df[series].dropna().unique().tolist())):
                rows = df[df[series] == series_value]
                lookup = rows.groupby(category, dropna=False)[value].sum().to_dict()
                values = [self._number(lookup.get(item, 0)) for item in category_values]
                all_values.extend(values)
                details = [
                    self._detail(
                        value,
                        f"{self._label(category)}: {label}",
                        f"{self._label(series)}: {self._display_value(series, series_value)}",
                    )
                    for label in labels
                ]
                datasets.append(self._dataset(
                    self._display_value(series, series_value), values, index, details
                ))
            return "bar", {"labels": labels, "datasets": datasets}, (
                self._label(category), self._label(value)
            ), all_values

        if len(measures) == 1 and len(dimensions) == 1:
            category, value = dimensions[0], measures[0]
            grouped = df.groupby(category, dropna=False)[value].sum().reset_index()
            labels = [self._display_value(category, item) for item in grouped[category]]
            values = [self._number(item) for item in grouped[value]]
            details = [self._detail(value, f"{self._label(category)}: {label}") for label in labels]
            if self._should_use_doughnut(grouped[value], value):
                dataset = self._dataset(self._label(value), values, 0, details, per_item=True)
                return "doughnut", {"labels": labels, "datasets": [dataset]}, ("", ""), values
            return "bar", {
                "labels": labels,
                "datasets": [self._dataset(self._label(value), values, 0, details)],
            }, (self._label(category), self._label(value)), values

        if dimensions and len(measures) >= 2:
            category = dimensions[0]
            grouped = df.groupby(category, dropna=False)[measures[:6]].sum().reset_index()
            labels = [self._display_value(category, item) for item in grouped[category]]
            datasets = []
            all_values: list[float] = []
            for index, column in enumerate(measures[:6]):
                values = [self._number(item) for item in grouped[column]]
                all_values.extend(values)
                details = [self._detail(column, f"{self._label(category)}: {label}") for label in labels]
                datasets.append(self._dataset(self._label(column), values, index, details))
            return "bar", {"labels": labels, "datasets": datasets}, (
                self._label(category), "값"
            ), all_values

        if len(measures) == 2:
            first, second = measures
            points = [
                {"x": self._number(x), "y": self._number(y)}
                for x, y in zip(df[first], df[second])
            ]
            values = [point["y"] for point in points]
            details = [self._detail(second, f"{self._label(first)}: {point['x']:,}") for point in points]
            dataset = self._dataset(self._label(second), points, 0, details)
            dataset.update({"showLine": False, "pointRadius": 5, "pointHoverRadius": 8})
            return "scatter", {"datasets": [dataset]}, (self._label(first), self._label(second)), values

        if len(measures) == 1:
            value = measures[0]
            labels = [str(index + 1) for index in range(len(df))]
            values = [self._number(item) for item in df[value]]
            details = [self._detail(value, f"순번: {label}") for label in labels]
            return "line", {
                "labels": labels,
                "datasets": [self._dataset(self._label(value), values, 0, details, line=True)],
            }, ("순번", self._label(value)), values

        if len(categorical) == 1:
            column = categorical[0]
            counts = df[column].fillna("미상").value_counts().head(10)
            labels = [self._display_value(column, item) for item in counts.index]
            values = [int(item) for item in counts.values]
            details = [self._detail(column, f"{self._label(column)}: {label}") for label in labels]
            return "bar", {
                "labels": labels,
                "datasets": [self._dataset("건수", values, 0, details)],
            }, (self._label(column), "건수"), values

        raise ValueError("Cannot determine appropriate visualization for this DataFrame")

    def _dataset(
        self,
        label: str,
        values: list[Any],
        color_index: int,
        details: list[dict[str, str]],
        *,
        per_item: bool = False,
        line: bool = False,
    ) -> dict[str, Any]:
        color = self.COLOR_PALETTE[color_index % len(self.COLOR_PALETTE)]
        colors: str | list[str] = color
        if per_item:
            colors = [self.COLOR_PALETTE[index % len(self.COLOR_PALETTE)] for index in range(len(values))]
        result = {
            "label": label,
            "data": values,
            "backgroundColor": colors if not line else "rgba(181,18,27,0.10)",
            "borderColor": color,
            "borderWidth": 2 if line else 0,
            "details": details,
        }
        if line:
            result.update({"tension": 0.28, "fill": False, "pointBackgroundColor": color})
        return result

    def _options(
        self,
        title: str,
        chart_type: str,
        axis_titles: tuple[str, str],
        values: list[float],
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "responsive": True,
            "maintainAspectRatio": False,
            "interaction": {"mode": "nearest", "intersect": True},
            "plugins": {
                "title": {"display": True, "text": title, "color": self.TEXT_COLOR, "font": {"size": 18}},
                "legend": {"display": True, "position": "top", "labels": {"color": self.TEXT_COLOR, "usePointStyle": True}},
                "tooltip": {"enabled": True, "displayColors": True},
            },
            "animation": {"duration": 450},
        }
        if chart_type != "doughnut":
            x_title, y_title = axis_titles
            options["scales"] = {
                "x": {"grid": {"display": False}, "border": {"display": False}, "title": {"display": bool(x_title), "text": x_title, "color": self.TEXT_COLOR}, "ticks": {"color": self.MUTED_TEXT_COLOR}},
                "y": {"grid": {"display": False}, "border": {"display": False}, "beginAtZero": True, "title": {"display": bool(y_title), "text": y_title, "color": self.TEXT_COLOR}, "ticks": {"color": self.MUTED_TEXT_COLOR}},
            }
            focused = self._focused_range(values)
            if focused:
                lower, upper, step = focused
                options["scales"]["y"].update({"beginAtZero": False, "min": lower, "max": upper, "ticks": {"color": self.MUTED_TEXT_COLOR, "stepSize": step}})
        return options

    @classmethod
    def _focused_range(cls, values: list[float]) -> tuple[float, float, float] | None:
        finite = [value for value in values if math.isfinite(value)]
        if len(finite) < 2:
            return None
        minimum, maximum = min(finite), max(finite)
        spread = maximum - minimum
        if minimum <= 0 or maximum < 1_000 or spread <= 0 or spread / maximum > 0.25:
            return None
        step = cls._nice_step(spread / 4)
        padding = max(spread * 0.2, step)
        lower = max(0, math.floor((minimum - padding) / step) * step)
        upper = math.ceil((maximum + padding) / step) * step
        return lower, upper, step

    @staticmethod
    def _nice_step(raw_step: float) -> float:
        exponent = math.floor(math.log10(raw_step))
        fraction = raw_step / (10 ** exponent)
        return next(value for value in (1, 2, 5, 10) if fraction <= value) * (10 ** exponent)

    def _detail(self, column: Any, *context: str) -> dict[str, str]:
        return {"name": self._label(column), "context": " · ".join(context), "description": self._description(column)}

    def _label(self, column: Any) -> str:
        name = str(column)
        return self.column_metadata.get(name, {}).get("logical_name") or name.replace("_", " ").title()

    def _description(self, column: Any) -> str:
        return self.column_metadata.get(str(column), {}).get("description") or ""

    def _is_dimension(self, column: Any) -> bool:
        metadata = self.column_metadata.get(str(column), {})
        role = str(metadata.get("sql_role") or "").lower()
        semantic_type = str(metadata.get("semantic_type") or "").lower()
        name = str(column).lower()
        return role in {"time", "code", "dimension", "identifier"} or semantic_type in {"year", "year_month", "date", "gender", "category", "code"} or any(token in name for token in ("_yr", "year", "_date", "_dt"))

    def _is_time_dimension(self, column: Any) -> bool:
        metadata = self.column_metadata.get(str(column), {})
        return str(metadata.get("sql_role") or "").lower() == "time" or str(metadata.get("semantic_type") or "").lower() in {"year", "year_month", "date"}

    def _display_value(self, column: Any, value: Any) -> str:
        metadata = self.column_metadata.get(str(column), {})
        raw = str(value)
        if str(metadata.get("semantic_type") or "").lower() == "gender" or str(metadata.get("code_name") or "").upper() == "SEX_TYPE":
            return {"F": "여성", "M": "남성"}.get(raw.upper(), raw)
        if self._is_time_dimension(column) and raw.removesuffix(".0").isdigit():
            digits = raw.removesuffix(".0")
            if len(digits) == 4:
                return f"{digits}년"
        return raw

    @staticmethod
    def _number(value: Any) -> float | int:
        number = float(value)
        return int(number) if number.is_integer() else number

    @staticmethod
    def _should_use_doughnut(values: pd.Series, column: Any) -> bool:
        name = str(column).lower()
        total = float(values.fillna(0).sum())
        return len(values) <= 8 and values.ge(0).all() and (any(token in name for token in ("percent", "percentage", "ratio", "rate", "share", "pct")) or 99 <= total <= 101)
