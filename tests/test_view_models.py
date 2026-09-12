from __future__ import annotations

from copy import deepcopy

import pandas as pd

from argus.app.view_models import (
    build_queue_view,
    display_pattern,
    display_queue,
    filter_queue_view,
    sort_queue_view,
)


def _queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "ARG-0002",
                "priority": "medium",
                "major_pattern": "rapid_movement",
                "transaction_count": 2,
                "total_flow": 150.0,
            },
            {
                "case_id": "ARG-0001",
                "priority": "high",
                "major_pattern": "fan_out",
                "transaction_count": 1,
                "total_flow": 300.0,
            },
        ]
    )


def _cases() -> dict[str, dict[str, object]]:
    return {
        "ARG-0001": {
            "case_id": "ARG-0001",
            "status": "pending_human_review",
            "transactions": [
                {
                    "transaction_id": "TX-1",
                    "from_node_id": "BANK-A::ACCOUNT-101",
                    "to_node_id": "BANK-B::ACCOUNT-202",
                    "timestamp": "2022-09-02T09:30:00Z",
                    "currency": "USD",
                }
            ],
        },
        "ARG-0002": {
            "case_id": "ARG-0002",
            "status": "pending_human_review",
            "transactions": [
                {
                    "transaction_id": "TX-2",
                    "from_node_id": "BANK-C::ACCOUNT-303",
                    "to_node_id": "BANK-D::ACCOUNT-404",
                    "timestamp": "2022-09-01T09:30:00Z",
                    "currency": "EUR",
                },
                {
                    "transaction_id": "TX-3",
                    "from_node_id": "BANK-D::ACCOUNT-404",
                    "to_node_id": "BANK-E::ACCOUNT-505",
                    "timestamp": "2022-09-03T09:30:00Z",
                    "currency": "GBP",
                },
            ],
        },
    }


def test_queue_projection_is_searchable_and_does_not_mutate_sources() -> None:
    queue = _queue()
    cases = _cases()
    queue_before = queue.copy(deep=True)
    cases_before = deepcopy(cases)

    view = build_queue_view(queue, cases)

    assert filter_queue_view(view, query="account-202")["case_id"].tolist() == ["ARG-0001"]
    assert filter_queue_view(view, query="ARG-0002")["case_id"].tolist() == ["ARG-0002"]
    assert filter_queue_view(view, query="fan out")["case_id"].tolist() == ["ARG-0001"]
    pd.testing.assert_frame_equal(queue, queue_before)
    assert cases == cases_before


def test_queue_projection_uses_session_status_and_honest_currency_context() -> None:
    view = build_queue_view(
        _queue(),
        _cases(),
        {"ARG-0001": {"status": "escalated"}},
    )

    first = view.set_index("case_id").loc["ARG-0001"]
    second = view.set_index("case_id").loc["ARG-0002"]
    assert first["status_label"] == "Escalated"
    assert first["currency_context"] == "USD"
    assert second["currency_context"] == "Mixed currencies"
    assert "Mixed currencies" in display_queue(view).loc[0, "Total Flow"]
    assert "Review Signal" in display_queue(view).columns
    assert "Primary Pattern" not in display_queue(view).columns


def test_queue_sorting_and_filters_are_stable() -> None:
    view = build_queue_view(_queue(), _cases())

    assert sort_queue_view(view, "Priority")["case_id"].tolist() == ["ARG-0001", "ARG-0002"]
    assert sort_queue_view(view, "Latest activity")["case_id"].tolist() == [
        "ARG-0002",
        "ARG-0001",
    ]
    assert filter_queue_view(view, priorities=["High"])["case_id"].tolist() == ["ARG-0001"]


def test_priority_sort_distinguishes_high_elevated_and_medium() -> None:
    queue = pd.DataFrame(
        [
            {"case_id": "MED", "priority": "medium"},
            {"case_id": "ELEV", "priority": "elevated"},
            {"case_id": "HIGH", "priority": "high"},
        ]
    )
    cases = {
        case_id: {"case_id": case_id, "status": "pending_review", "transactions": []}
        for case_id in queue["case_id"]
    }

    view = build_queue_view(queue, cases)

    assert sort_queue_view(view, "Priority")["case_id"].tolist() == ["HIGH", "ELEV", "MED"]


def test_empty_queue_projection_keeps_display_contract() -> None:
    view = build_queue_view(pd.DataFrame(columns=["case_id"]), {})

    assert view.empty
    assert display_queue(view).empty


def test_graphsage_score_label_is_a_review_signal_not_a_behavioral_pattern() -> None:
    assert display_pattern("high_graphsage_transaction_score") == "Elevated network ranking"
    assert display_pattern("fan_out") == "Fan Out"
