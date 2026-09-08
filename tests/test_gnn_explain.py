from __future__ import annotations

import pytest
import torch

from argus.gnn import (
    GraphSAGEConfig,
    GraphSAGEEdgeClassifier,
    GraphSAGEModelError,
    build_directed_mean_adjacency,
    explain_edge_inputs,
)


def _transparent_model() -> tuple[GraphSAGEEdgeClassifier, object]:
    config = GraphSAGEConfig(
        num_nodes=3,
        transaction_feature_dim=1,
        node_input_dim=1,
        hidden_dim=1,
        embedding_dim=1,
        num_layers=1,
        classifier_hidden_dim=1,
    )
    model = GraphSAGEEdgeClassifier(config)
    adjacency = build_directed_mean_adjacency(torch.tensor([[0, 1], [1, 2]]), num_nodes=3)
    with torch.no_grad():
        model.node_inputs.weight[:, 0] = torch.tensor([1.0, 2.0, 4.0])
        model.encoder.layers[0].linear.weight[:] = torch.tensor([[1.0, 0.0, 0.0]])
        model.encoder.layers[0].linear.bias.zero_()
        model.edge_classifier[0].weight[:] = torch.tensor([[1.0, 2.0, 3.0]])
        model.edge_classifier[0].bias[:] = 10.0
        model.edge_classifier[3].weight[:] = 1.0
        model.edge_classifier[3].bias.zero_()
    return model, adjacency


def test_local_attribution_exposes_role_specific_gradient_times_input() -> None:
    model, adjacency = _transparent_model()
    model.train()

    attribution = explain_edge_inputs(
        model,
        adjacency,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([[5.0]]),
    )

    assert attribution.logits.item() == pytest.approx(30.0)
    assert attribution.sender_gradients.item() == pytest.approx(1.0)
    assert attribution.receiver_gradients.item() == pytest.approx(2.0)
    assert attribution.transaction_gradients.item() == pytest.approx(3.0)
    assert attribution.sender_gradient_x_input.item() == pytest.approx(1.0)
    assert attribution.receiver_gradient_x_input.item() == pytest.approx(4.0)
    assert attribution.transaction_gradient_x_input.item() == pytest.approx(15.0)
    assert attribution.probabilities.item() == pytest.approx(
        torch.sigmoid(torch.tensor(30.0)).item()
    )
    assert model.training is True


def test_negative_class_attribution_reverses_gradient_direction() -> None:
    model, adjacency = _transparent_model()
    positive = explain_edge_inputs(
        model,
        adjacency,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([[5.0]]),
        target_class=1,
    )
    negative = explain_edge_inputs(
        model,
        adjacency,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([[5.0]]),
        target_class=0,
    )

    assert torch.equal(negative.transaction_gradients, -positive.transaction_gradients)
    assert torch.equal(negative.sender_gradients, -positive.sender_gradients)
    assert torch.equal(negative.receiver_gradients, -positive.receiver_gradients)


def test_attribution_rejects_non_binary_target_class() -> None:
    model, adjacency = _transparent_model()

    with pytest.raises(GraphSAGEModelError, match="target_class"):
        explain_edge_inputs(
            model,
            adjacency,
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([[5.0]]),
            target_class=2,
        )
