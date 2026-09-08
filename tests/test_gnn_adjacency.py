from __future__ import annotations

import pytest
import torch

from argus.gnn import (
    DirectedMeanAdjacency,
    GraphAdjacencyError,
    build_directed_mean_adjacency,
)


def test_directed_means_preserve_direction_and_repeated_transfer_weight() -> None:
    # A->B occurs twice, C->B once, and B->C once; D is isolated.
    edge_index = torch.tensor([[0, 0, 2, 1], [1, 1, 1, 2]], dtype=torch.long)
    adjacency = build_directed_mean_adjacency(edge_index, num_nodes=4)
    states = torch.tensor([[1.0], [3.0], [5.0], [7.0]])

    inbound, outbound = adjacency.aggregate(states)

    assert adjacency.edge_count == 4
    assert adjacency.incoming.layout == torch.sparse_coo
    assert adjacency.outgoing.layout == torch.sparse_coo
    assert inbound[:, 0].tolist() == pytest.approx([0.0, 7.0 / 3.0, 3.0, 0.0])
    assert outbound[:, 0].tolist() == pytest.approx([3.0, 5.0, 3.0, 0.0])


def test_empty_graph_returns_zero_means() -> None:
    adjacency = build_directed_mean_adjacency(
        torch.empty((2, 0), dtype=torch.long), num_nodes=3, dtype=torch.float64
    )
    inbound, outbound = adjacency.aggregate(torch.ones((3, 2), dtype=torch.float64))

    assert torch.equal(inbound, torch.zeros((3, 2), dtype=torch.float64))
    assert torch.equal(outbound, torch.zeros((3, 2), dtype=torch.float64))


def test_directed_adjacency_checkpoint_round_trip() -> None:
    original = build_directed_mean_adjacency(
        torch.tensor([[0, 2, 0], [1, 1, 1]]), num_nodes=3, dtype=torch.float64
    )

    restored = DirectedMeanAdjacency.from_state_dict(original.state_dict())

    states = torch.tensor([[1.5], [2.5], [4.5]], dtype=torch.float64)
    original_means = original.aggregate(states)
    restored_means = restored.aggregate(states)
    assert restored.edge_count == original.edge_count
    assert all(
        torch.equal(left, right) for left, right in zip(original_means, restored_means, strict=True)
    )


@pytest.mark.parametrize(
    ("edge_index", "message"),
    [
        (torch.tensor([0, 1]), "shape"),
        (torch.tensor([[0.0], [1.0]]), "integer dtype"),
        (torch.tensor([[0], [3]]), "must lie"),
        (torch.tensor([[0], [-1]]), "must lie"),
    ],
)
def test_directed_adjacency_rejects_malformed_edges(edge_index: torch.Tensor, message: str) -> None:
    with pytest.raises(GraphAdjacencyError, match=message):
        build_directed_mean_adjacency(edge_index, num_nodes=3)


def test_adjacency_rejects_mismatched_state_dtype() -> None:
    adjacency = build_directed_mean_adjacency(torch.tensor([[0], [1]]), num_nodes=2)

    with pytest.raises(GraphAdjacencyError, match="share the adjacency"):
        adjacency.aggregate(torch.ones((2, 1), dtype=torch.float64))
