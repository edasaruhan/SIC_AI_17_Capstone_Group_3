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
                {"id": "BANK-A::A1", "role": "sender", "is_seed": True},
                {"id": "BANK-B::B1", "role": "receiver", "is_seed": True},
                {"id": "BANK-C::C1", "role": "neighbor"},
            ],
            "edges": [
                {
                    "source": "BANK-A::A1",
                    "target": "BANK-B::B1",
                    "transaction_id": "TX-1",
                    "amount": 100.0,
                    "timestamp": "2022-09-01T00:02:00Z",
                    "is_focal": True,
                },
                {
                    "source": "BANK-B::B1",
                    "target": "BANK-C::C1",
                    "transaction_id": "TX-2",
                    "amount": 90.0,
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
    assert figure.data[0].line.color == "#f59e0b"


def test_timeline_uses_saved_edge_timestamps_and_amounts() -> None:
    figure = build_timeline_figure(_visual_case())

    assert figure is not None
    assert list(figure.data[0].y) == [100.0, 90.0]
    assert list(figure.data[0].text) == ["TX-1", "TX-2"]
    assert list(figure.data[0].marker.color) == ["#f59e0b", "#0f766e"]
    assert figure.layout.xaxis.title.text == "Timestamp (UTC)"


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

    curves = pd.DataFrame(
        [
            {"model": "GraphSAGE", "recall": 1.0, "precision": 0.001},
            {"model": "GraphSAGE", "recall": 0.2, "precision": 0.4},
        ]
    )
    curve = build_pr_curve_figure(curves)
    assert list(curve.data[0].x) == [0.2, 1.0]
    assert list(curve.data[0].y) == [0.4, 0.001]
