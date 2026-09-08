from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argus.sprint4.graph_data import (
    NodeFeatureNormalizer,
    Sprint4DataError,
    build_graph_view,
    endpoint_indices,
)


def _context() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "from_node_id": ["001::A", "001::A", "002::B"],
            "to_node_id": ["002::B", "002::B", "003::C"],
            "amount_paid": [10.0, 20.0, 5.0],
        }
    )


def test_graph_view_preserves_direction_and_repeated_edges() -> None:
    view = build_graph_view(_context(), required_node_ids=("004::D",))
    assert view.context_edge_rows == 3
    assert view.node_count == 4
    source = view.node_to_index["001::A"]
    target = view.node_to_index["002::B"]
    assert np.count_nonzero((view.edge_sources == source) & (view.edge_targets == target)) == 2
    assert view.node_features.shape == (4, 6)
    assert np.isfinite(view.node_features).all()


def test_graph_view_applies_frozen_normalizer_to_later_nodes() -> None:
    fitted = build_graph_view(_context(), required_node_ids=())
    later = build_graph_view(
        pd.DataFrame(
            {
                "from_node_id": ["001::A"],
                "to_node_id": ["009::Z"],
                "amount_paid": [100.0],
            }
        ),
        required_node_ids=("010::NEW",),
        normalizer=fitted.normalizer,
    )
    assert later.normalizer == fitted.normalizer
    assert np.isfinite(later.node_features).all()


def test_graph_view_does_not_aggregate_cross_currency_amounts() -> None:
    first = _context()
    second = _context()
    second["amount_paid"] = [1e12, 0.01, 999_999.0]

    first_view = build_graph_view(first, required_node_ids=())
    second_view = build_graph_view(second, required_node_ids=())

    np.testing.assert_array_equal(first_view.node_features, second_view.node_features)


def test_endpoint_indices_rejects_nodes_missing_from_view() -> None:
    view = build_graph_view(_context(), required_node_ids=())
    frame = pd.DataFrame({"from_node_id": ["001::A"], "to_node_id": ["999::X"]})
    with pytest.raises(Sprint4DataError, match="omitted"):
        endpoint_indices(frame, view.node_to_index)


def test_node_normalizer_rejects_nonpositive_scale() -> None:
    with pytest.raises(Sprint4DataError, match="positive"):
        NodeFeatureNormalizer(means=(0.0, 0.0), scales=(1.0, 0.0))


def test_graph_view_rejects_noncomposite_node_identity() -> None:
    context = _context()
    context.loc[0, "from_node_id"] = "not-composite"
    with pytest.raises(Sprint4DataError, match="composite"):
        build_graph_view(context, required_node_ids=())
