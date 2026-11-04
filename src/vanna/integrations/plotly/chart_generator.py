"""Responsive Plotly chart generator using the GenSQL dashboard style."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


class PlotlyChartGenerator:
    """Build an interactive Plotly figure from a SQL-result DataFrame."""

    COLOR_PALETTE = ["#B5121B", "#CF2E2E", "#E34A3F", "#E96B57", "#F08D75", "#F3B09E"]
    TEXT_COLOR = "#242424"
    MUTED_TEXT_COLOR = "#6B6B6B"
    GRID_COLOR = "#DCE3EA"

    def generate_chart(
        self,
        df: pd.DataFrame,
        title: str = "Chart",
        column_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if df.empty:
            raise ValueError("Cannot visualize empty DataFrame")
        frame = self._infer_datetimes(df.copy())
        self.column_metadata = column_metadata or {}
        figure, chart_type = self._create_figure(frame)
        self._apply_dashboard_style(figure, title, chart_type)
        return {
            "chart_type": chart_type,
            "figure": json.loads(pio.to_json(figure)),
            "config": {
                "responsive": True,
                "displaylogo": False,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        }

    @staticmethod
    def _infer_datetimes(df: pd.DataFrame) -> pd.DataFrame:
        for column in df.select_dtypes(include=["object"]).columns:
            name = str(column).lower()
            if not any(token in name for token in ("date", "time", "month", "year")):
                continue
            converted = pd.to_datetime(df[column], errors="coerce")
            if converted.notna().mean() >= 0.8:
                df[column] = converted
        return df

    def _create_figure(self, df: pd.DataFrame) -> tuple[go.Figure, str]:
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetimes = df.select_dtypes(include=["datetime64"]).columns.tolist()
        numeric_dimensions = [column for column in numeric if self._is_dimension(column)]
        measures = [column for column in numeric if column not in numeric_dimensions]

        if len(df) == 1 and measures:
            figure = go.Figure()
            values = measures[:4]
            for index, column in enumerate(values):
                figure.add_trace(go.Indicator(
                    mode="number", value=float(df.iloc[0][column]),
                    number={"valueformat": ",", "font": {"color": self.TEXT_COLOR}},
                    title={"text": str(column).replace("_", " ").title(), "font": {"color": self.MUTED_TEXT_COLOR, "size": 12}},
                    domain={"row": 0, "column": index},
                ))
            figure.update_layout(grid={"rows": 1, "columns": len(values), "pattern": "independent"})
            return figure, "kpi"

        if datetimes and measures:
            time_column = datetimes[0]
            ordered = df.sort_values(time_column)
            figure = go.Figure()
            for index, column in enumerate(measures[:5]):
                figure.add_trace(go.Scatter(
                        x=ordered[time_column], y=ordered[column], mode="lines+markers+text", name=self._label(column),
                        text=[self._format_value(value) for value in ordered[column]], textposition="top center",
                        customdata=[[self._description(column)] for _ in range(len(ordered))],
                        hovertemplate=f"<b>{self._label(column)}</b><br>{self._label(time_column)}: %{{x}}<br>값: %{{y:,}}<br>%{{customdata[0]}}<extra></extra>",
                    line={"color": self.COLOR_PALETTE[index % len(self.COLOR_PALETTE)], "width": 2},
                    fill="tozeroy" if index == 0 else None,
                    fillcolor="rgba(181,18,27,0.10)" if index == 0 else None,
                ))
            figure.update_layout(xaxis_title=self._label(time_column), yaxis_title="값", hovermode="closest")
            return figure, "line"

        if len(measures) == 1 and not categorical and not numeric_dimensions:
            column = measures[0]
            return go.Figure(go.Histogram(x=df[column], marker_color=self.COLOR_PALETTE[0], name=self._label(column), hovertemplate=f"<b>{self._label(column)}</b><br>구간: %{{x}}<br>건수: %{{y:,}}<br>{self._description(column)}<extra></extra>")), "histogram"

        dimensions = categorical + numeric_dimensions

        if len(measures) == 1 and len(dimensions) >= 2:
            value = measures[0]
            time_dimensions = [column for column in dimensions if self._is_time_dimension(column)]
            series = time_dimensions[0] if time_dimensions else dimensions[1]
            category = next(column for column in dimensions if column != series)
            grouped = df.groupby([category, series], dropna=False)[value].sum().reset_index()
            figure = go.Figure()
            for index, (series_value, rows) in enumerate(grouped.groupby(series, dropna=False, sort=True)):
                series_label = self._display_value(series, series_value)
                figure.add_trace(go.Bar(
                    x=[self._display_value(category, item) for item in rows[category]],
                    y=rows[value].tolist(),
                    name=series_label,
                    marker_color=self.COLOR_PALETTE[index % len(self.COLOR_PALETTE)],
                    text=[self._format_value(item) for item in rows[value]],
                    textposition="outside",
                    customdata=[
                        [self._label(category), self._display_value(category, item), self._label(value), self._description(value)]
                        for item in rows[category]
                    ],
                    hovertemplate=(
                        f"<b>{self._label(value)}</b><br>"
                        "%{customdata[0]}: %{customdata[1]}<br>"
                        f"{self._label(series)}: {series_label}<br>"
                        "%{customdata[2]}: %{y:,}<br>%{customdata[3]}<extra></extra>"
                    ),
                ))
            figure.update_layout(
                barmode="group",
                xaxis_title=self._label(category),
                yaxis_title=self._label(value),
            )
            return figure, "grouped_bar"

        if len(measures) == 1 and len(dimensions) == 1:
            category, value = dimensions[0], measures[0]
            grouped = df.groupby(category, dropna=False)[value].sum().reset_index()
            descriptions = [self._description(value)] * len(grouped)
            if self._should_use_donut(grouped[value], value):
                figure = go.Figure(go.Pie(
                    labels=[self._display_value(category, item) for item in grouped[category]], values=grouped[value], hole=0.58,
                    marker={"colors": self.COLOR_PALETTE, "line": {"color": "white", "width": 2}},
                    textinfo="percent+label", textposition="outside", customdata=descriptions,
                    hovertemplate=f"<b>%{{label}}</b><br>{self._label(value)}: %{{value:,}}<br>비율: %{{percent}}<br>%{{customdata}}<extra></extra>",
                ))
                return figure, "donut"
            figure = go.Figure(go.Bar(
                x=[self._display_value(category, item) for item in grouped[category]], y=grouped[value], marker_color=self.COLOR_PALETTE[0],
                name=self._label(value), text=[self._format_value(item) for item in grouped[value]], textposition="outside",
                customdata=[[item] for item in descriptions],
                hovertemplate=f"<b>{self._label(value)}</b><br>{self._label(category)}: %{{x}}<br>값: %{{y:,}}<br>%{{customdata[0]}}<extra></extra>",
            ))
            figure.update_layout(xaxis_title=self._label(category), yaxis_title=self._label(value))
            return figure, "bar"

        if categorical and len(measures) >= 2:
            category = categorical[0]
            grouped = df.groupby(category, dropna=False)[measures[:6]].sum().reset_index()
            figure = go.Figure()
            for index, column in enumerate(measures[:6]):
                figure.add_trace(go.Bar(
                    x=[self._display_value(category, item) for item in grouped[category]], y=grouped[column], name=self._label(column),
                    marker_color=self.COLOR_PALETTE[index % len(self.COLOR_PALETTE)],
                    text=[self._format_value(item) for item in grouped[column]], textposition="outside",
                    customdata=[[self._description(column)] for _ in range(len(grouped))],
                    hovertemplate=f"<b>{self._label(column)}</b><br>{self._label(category)}: %{{x}}<br>값: %{{y:,}}<br>%{{customdata[0]}}<extra></extra>",
                ))
            figure.update_layout(barmode="group", xaxis_title=self._label(category), yaxis_title="값")
            return figure, "grouped_bar"

        if len(measures) == 2:
            first, second = measures
            figure = go.Figure(go.Scatter(
                x=df[first], y=df[second], mode="markers+text", name=self._label(second),
                text=[self._format_value(item) for item in df[second]], textposition="top center",
                marker={"color": self.COLOR_PALETTE[0], "size": 9, "opacity": 0.82, "line": {"color": "white", "width": 1}},
                customdata=[[self._description(second)] for _ in range(len(df))],
                hovertemplate=f"<b>{self._label(second)}</b><br>{self._label(first)}: %{{x:,}}<br>{self._label(second)}: %{{y:,}}<br>%{{customdata[0]}}<extra></extra>",
            ))
            figure.update_layout(xaxis_title=self._label(first), yaxis_title=self._label(second))
            return figure, "scatter"

        if len(measures) >= 3:
            correlation = df[measures].corr()
            labels = [self._label(column) for column in measures]
            figure = go.Figure(go.Heatmap(z=correlation.values, x=labels, y=labels, zmin=-1, zmax=1, texttemplate="%{z:.2f}", colorscale=[[0, "#F3B09E"], [0.5, "#FFF7F5"], [1, "#B5121B"]], colorbar={"title": "상관계수"}, hovertemplate="<b>%{x} × %{y}</b><br>상관계수: %{z:.3f}<extra></extra>"))
            return figure, "heatmap"

        if len(categorical) >= 2:
            first, second = categorical[:2]
            grouped = df.groupby([first, second], dropna=False).size().reset_index(name="count")
            figure = go.Figure()
            for index, (name, values) in enumerate(grouped.groupby(second, dropna=False)):
                figure.add_trace(go.Bar(x=values[first].astype(str), y=values["count"], name=str(name), marker_color=self.COLOR_PALETTE[index % len(self.COLOR_PALETTE)], text=values["count"], textposition="outside"))
            figure.update_layout(barmode="group", xaxis_title=self._label(first), yaxis_title="건수")
            return figure, "grouped_bar"

        if len(categorical) == 1:
            column = categorical[0]
            counts = df[column].fillna("Unknown").astype(str).value_counts().head(10).sort_values()
            figure = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=self.COLOR_PALETTE[0]))
            figure.update_traces(text=counts.values, textposition="outside", name=self._label(column), hovertemplate=f"<b>%{{y}}</b><br>건수: %{{x:,}}<br>{self._description(column)}<extra></extra>")
            figure.update_layout(xaxis_title="건수", yaxis_title=self._label(column))
            return figure, "horizontal_bar"

        if len(df.columns) >= 2:
            first, second = df.columns[:2]
            values = pd.to_numeric(df[second], errors="coerce").fillna(0)
            figure = go.Figure(go.Bar(x=df[first].astype(str), y=values, marker_color=self.COLOR_PALETTE[1]))
            figure.update_traces(text=[self._format_value(item) for item in values], textposition="outside", name=self._label(second))
            figure.update_layout(xaxis_title=self._label(first), yaxis_title=self._label(second))
            return figure, "bar"

        raise ValueError("Cannot determine appropriate visualization for this DataFrame")

    def _apply_dashboard_style(self, figure: go.Figure, title: str, chart_type: str) -> None:
        figure.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 18, "color": self.TEXT_COLOR}},
            autosize=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Malgun Gothic, Inter, sans-serif", "color": self.TEXT_COLOR, "size": 12},
            colorway=self.COLOR_PALETTE, margin={"l": 56, "r": 24, "t": 64, "b": 56},
            hoverlabel={"bgcolor": "white", "font": {"color": self.TEXT_COLOR}},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5, "font": {"color": self.TEXT_COLOR}},
            uirevision="gensql-chart",
        )
        if chart_type != "kpi":
            figure.update_xaxes(showgrid=False, zeroline=False, showline=False, tickfont={"color": self.MUTED_TEXT_COLOR}, title_font={"color": self.MUTED_TEXT_COLOR}, automargin=True)
            figure.update_yaxes(showgrid=False, zeroline=False, showline=False, tickfont={"color": self.MUTED_TEXT_COLOR}, title_font={"color": self.MUTED_TEXT_COLOR}, tickformat="~s", automargin=True)
        if chart_type in {"bar", "grouped_bar", "line"}:
            self._focus_large_value_axis(figure)

    @classmethod
    def _focus_large_value_axis(cls, figure: go.Figure) -> None:
        """Expand small differences between large positive values with a rounded axis window."""
        values: list[float] = []
        for trace in figure.data:
            if trace.type not in {"bar", "scatter"} or getattr(trace, "y", None) is None:
                continue
            for value in trace.y:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    values.append(number)

        if len(values) < 2:
            return
        minimum, maximum = min(values), max(values)
        spread = maximum - minimum
        if minimum <= 0 or maximum < 1_000 or spread <= 0 or spread / maximum > 0.25:
            return

        step = cls._nice_step(spread / 4)
        padding = max(spread * 0.2, step)
        lower = math.floor((minimum - padding) / step) * step
        upper = math.ceil((maximum + padding) / step) * step
        lower = max(0, lower)
        if lower >= minimum or upper <= maximum:
            return
        figure.update_yaxes(range=[lower, upper], dtick=step, tickformat=",d")

    @staticmethod
    def _nice_step(raw_step: float) -> float:
        exponent = math.floor(math.log10(raw_step))
        fraction = raw_step / (10 ** exponent)
        nice_fraction = next(value for value in (1, 2, 5, 10) if fraction <= value)
        return nice_fraction * (10 ** exponent)

    def _label(self, column: Any) -> str:
        name = str(column)
        metadata = self.column_metadata.get(name, {})
        return metadata.get("logical_name") or name.replace("_", " ").title()

    def _description(self, column: Any) -> str:
        metadata = self.column_metadata.get(str(column), {})
        return metadata.get("description") or ""

    def _is_dimension(self, column: Any) -> bool:
        metadata = self.column_metadata.get(str(column), {})
        role = str(metadata.get("sql_role") or "").lower()
        semantic_type = str(metadata.get("semantic_type") or "").lower()
        name = str(column).lower()
        return role in {"time", "code", "dimension", "identifier"} or semantic_type in {
            "year", "year_month", "date", "gender", "category", "code"
        } or any(token in name for token in ("_yr", "year", "_date", "_dt"))

    def _is_time_dimension(self, column: Any) -> bool:
        metadata = self.column_metadata.get(str(column), {})
        return str(metadata.get("sql_role") or "").lower() == "time" or str(
            metadata.get("semantic_type") or ""
        ).lower() in {"year", "year_month", "date"}

    def _display_value(self, column: Any, value: Any) -> str:
        metadata = self.column_metadata.get(str(column), {})
        semantic_type = str(metadata.get("semantic_type") or "").lower()
        code_name = str(metadata.get("code_name") or "").upper()
        raw = str(value)
        if semantic_type == "gender" or code_name == "SEX_TYPE":
            return {"F": "여성", "M": "남성"}.get(raw.upper(), raw)
        if self._is_time_dimension(column) and raw.removesuffix(".0").isdigit():
            digits = raw.removesuffix(".0")
            if len(digits) == 4:
                return f"{digits}년"
        return raw

    @staticmethod
    def _format_value(value: Any) -> str:
        if pd.isna(value):
            return "-"
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            return f"{value:,.2f}".rstrip("0").rstrip(".")
        return str(value)

    @staticmethod
    def _should_use_donut(values: pd.Series, column: Any) -> bool:
        name = str(column).lower()
        total = float(values.fillna(0).sum())
        return len(values) <= 8 and values.ge(0).all() and (
            any(token in name for token in ("percent", "percentage", "ratio", "rate", "share", "pct"))
            or 99 <= total <= 101
        )
