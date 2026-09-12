from __future__ import annotations

import pandas as pd

from argus.app.figures import (
    build_model_metric_figure,
    build_network_figure,
    build_pr_curve_figure,
    build_timeline_figure,
)


def _visual_case() -> dict[str, object]:
    return {
        "sender_id": "BANK-A::A1",
        "receiver_id": "BANK-B::B1",
        "network": {
            "nodes": [
                {
                    "id": "BANK-A::A1",
                    "role": "focal_sender",
                    "is_seed": True,
                    "case_in_degree": 0,
                    "case_out_degree": 1,
                },
                {
                    "id": "BANK-B::B1",
                    "role": "focal_receiver",
                    "is_seed": True,
                    "case_in_degree": 1,
                    "case_out_degree": 1,
                },
                {
                    "id": "BANK-C::C1",
                    "role": "strictly_prior_context",
                    "case_in_degree": 1,
                    "case_out_degree": 0,
                },
            ],
            "edges": [
                {
                    "source": "BANK-A::A1",
                    "target": "BANK-B::B1",
                    "transaction_id": "TX-1",
                    "amount": 100.0,
                    "currency": "USD",
                    "timestamp": "2022-09-01T00:02:00Z",
                    "is_focal": True,
                },
                {
                    "source": "BANK-B::B1",
                    "target": "BANK-C::C1",
                    "transaction_id": "TX-2",
                    "amount": 90.0,
                    "currency": "USD",
                    "timestamp": "2022-09-01T00:05:00Z",
                    "suspicious": False,
                },
            ],
        },
    }


def test_network_figure_preserves_directed_edges_with_arrowheads() -> None:
    figure = build_network_figure(_visual_case())

    assert figure is not None
    assert len(figure.layout.annotations) == 2
    assert all(annotation.showarrow for annotation in figure.layout.annotations)
    assert figure.layout.annotations[0].arrowhead == 3
    assert len(figure.data[-1].x) == 3
    assert "BANK-A::A1 → BANK-B::B1" in figure.data[0].hovertemplate
    assert figure.data[0].line.color == "#C97918"
    assert list(figure.data[0].customdata[0]) == [
        "TX-1",
        "100.00",
        "USD",
        "2022-09-01T00:02:00Z",
        "Focal transfer",
    ]
    for label in ("Transaction", "Amount", "Currency", "Timestamp", "Role"):
        assert label in figure.data[0].hovertemplate


def test_network_figure_has_readable_roles_and_plotly_navigation_controls() -> None:
    figure = build_network_figure(_visual_case())

    assert figure is not None
    legend_names = {trace.name for trace in figure.data if trace.showlegend}
    assert {
        "Focal transfer",
        "Earlier case context",
        "Focal sender",
        "Focal receiver",
        "Network context",
    }.issubset(legend_names)
    node_trace = figure.data[-1]
    assert list(node_trace.text) == ["Primary sender", "Selected receiver", "Linked account 1"]
    account_rows = {row[0]: row for row in node_trace.customdata}
    assert account_rows["BANK-A::A1"][1:] == ["Focal sender", 0, 1]
    assert account_rows["BANK-B::B1"][1:] == ["Focal receiver", 1, 1]
    assert list(node_trace.marker.symbol) == ["diamond", "square", "circle"]
    assert figure.layout.dragmode == "pan"
    assert figure.layout.xaxis.fixedrange is False
    assert figure.layout.yaxis.fixedrange is False
    assert set(figure.layout.modebar.add) >= {
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
    }


def test_network_figure_layout_is_deterministic() -> None:
    first = build_network_figure(_visual_case())
    second = build_network_figure(_visual_case())

    assert first is not None and second is not None
    assert list(first.data[-1].x) == list(second.data[-1].x)
    assert list(first.data[-1].y) == list(second.data[-1].y)
    assert [annotation.to_plotly_json() for annotation in first.layout.annotations] == [
        annotation.to_plotly_json() for annotation in second.layout.annotations
    ]


def test_timeline_uses_saved_edge_timestamps_and_amounts() -> None:
    figure = build_timeline_figure(_visual_case())

    assert figure is not None
    assert list(figure.data[0].y) == [100.0, 90.0]
    assert list(figure.data[0].text) == ["100 USD", "90 USD"]
    assert list(figure.data[0].marker.color) == ["#C97918", "#0B6B66"]
    assert list(figure.data[0].marker.symbol) == ["diamond", "circle"]
    assert list(figure.data[0].customdata[0]) == [
        "TX-1",
        "BANK-A::A1",
        "BANK-B::B1",
        "USD",
        "Focal transfer",
    ]
    assert figure.layout.xaxis.title.text == "Timestamp (UTC)"
    assert list(figure.layout.yaxis.range) == [87.5, 104.5]
    assert figure.data[0].mode == "lines+markers+text"
    assert figure.layout.dragmode == "pan"


def test_figures_return_none_when_case_has_no_graph_or_timeline() -> None:
    assert build_network_figure({"case_id": "ARG-1"}) is None
    assert build_timeline_figure({"case_id": "ARG-1"}) is None


def test_model_figures_use_only_supplied_saved_values() -> None:
    comparison = pd.DataFrame(
        [
            {"model": "LightGBM", "pr_auc": 0.47, "fpr": 0.005},
            {"model": "GraphSAGE", "pr_auc": 0.31, "fpr": 0.007},
        ]
    )
    bars = build_model_metric_figure(comparison)
    assert list(bars.data[0].y) == [0.47, 0.31]
    assert list(bars.data[1].y) == [0.005, 0.007]
    assert bars.data[0].marker.color == "#0B6B66"
    assert bars.data[1].marker.color == "#C97918"
    assert list(bars.data[0].x) == [
        "LightGBM",
        "GraphSAGE (research comparator)",
    ]

    curves = pd.DataFrame(
        [
            {"model": "GraphSAGE", "recall": 1.0, "precision": 0.001},
            {"model": "GraphSAGE", "recall": 0.2, "precision": 0.4},
        ]
    )
    curve = build_pr_curve_figure(curves)
    assert list(curve.data[0].x) == [0.2, 1.0]
    assert list(curve.data[0].y) == [0.4, 0.001]
    assert curve.data[0].name == "GraphSAGE (research comparator)"
    assert curve.data[0].line.color == "#73838F"
    assert curve.data[0].line.dash == "dot"
