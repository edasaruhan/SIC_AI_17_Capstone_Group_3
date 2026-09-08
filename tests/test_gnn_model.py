from __future__ import annotations

import random

import pytest
import torch

from argus.gnn import (
    GraphSAGEConfig,
    GraphSAGEEdgeClassifier,
    GraphSAGEModelError,
    build_directed_mean_adjacency,
    configure_deterministic_cpu,
    predict_edge_scores,
)


def _model_inputs() -> tuple[
    GraphSAGEEdgeClassifier,
    object,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    config = GraphSAGEConfig(
        num_nodes=4,
        transaction_feature_dim=3,
        node_input_dim=4,
        hidden_dim=5,
        embedding_dim=2,
        classifier_hidden_dim=6,
        dropout=0.0,
        seed=17,
    )
    model = GraphSAGEEdgeClassifier(config)
    adjacency = build_directed_mean_adjacency(
        torch.tensor([[0, 1, 1, 2], [1, 0, 2, 3]]), num_nodes=4
    )
    senders = torch.tensor([0, 1, 2], dtype=torch.long)
    receivers = torch.tensor([1, 2, 3], dtype=torch.long)
    features = torch.tensor([[1.0, 0.5, -1.0], [0.0, 2.0, 0.5], [3.0, -0.5, 1.0]])
    return model, adjacency, senders, receivers, features


def test_model_forward_and_batched_prediction_return_raw_and_probability_scores() -> None:
    model, adjacency, senders, receivers, features = _model_inputs()
    model.train()

    logits = model(adjacency, senders, receivers, features)
    predictions = predict_edge_scores(model, adjacency, senders, receivers, features, batch_size=2)

    assert logits.shape == (3,)
    assert torch.allclose(predictions.logits, logits)
    assert torch.allclose(predictions.probabilities, torch.sigmoid(logits))
    assert bool(torch.all((predictions.probabilities >= 0) & (predictions.probabilities <= 1)))
    assert model.training is True


def test_edge_classifier_combines_sender_receiver_and_transaction_inputs() -> None:
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

    logits = model(
        adjacency,
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
        torch.tensor([[5.0], [7.0]]),
    )

    assert logits.tolist() == pytest.approx([30.0, 41.0])


def test_model_seed_is_reproducible_without_consuming_caller_rng() -> None:
    config = GraphSAGEConfig(num_nodes=3, transaction_feature_dim=2, seed=91)
    torch.manual_seed(1234)
    expected_after_construction = torch.rand(3)
    torch.manual_seed(1234)

    first = GraphSAGEEdgeClassifier(config)
    actual_after_construction = torch.rand(3)
    second = GraphSAGEEdgeClassifier(config)

    assert torch.equal(expected_after_construction, actual_after_construction)
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )


def test_checkpoint_round_trip_preserves_config_and_scores() -> None:
    model, adjacency, senders, receivers, features = _model_inputs()
    before = predict_edge_scores(model, adjacency, senders, receivers, features)

    checkpoint = model.export_checkpoint()
    restored = GraphSAGEEdgeClassifier.from_checkpoint(checkpoint)
    after = predict_edge_scores(restored, adjacency, senders, receivers, features)

    assert checkpoint["format"] == "argus.graphsage_edge_classifier.v1"
    assert restored.config == model.config
    assert torch.equal(before.logits, after.logits)
    assert torch.equal(before.probabilities, after.probabilities)
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["state_dict"].values())


def test_explicit_node_features_support_inductive_different_sized_graph() -> None:
    model, _, _, _, _ = _model_inputs()  # Learned input table contains four nodes.
    inference_adjacency = build_directed_mean_adjacency(
        torch.tensor([[0, 4, 2, 3], [4, 1, 3, 0]]), num_nodes=5
    )
    node_features = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
    senders = torch.tensor([4, 3], dtype=torch.long)
    receivers = torch.tensor([1, 0], dtype=torch.long)
    transaction_features = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])

    predictions = predict_edge_scores(
        model,
        inference_adjacency,
        senders,
        receivers,
        transaction_features,
        node_features=node_features,
    )
    embeddings = model.encode_nodes(inference_adjacency, node_features=node_features)
    direct_logits = model.classify_edges(embeddings, senders, receivers, transaction_features)

    assert embeddings.shape == (5, model.config.embedding_dim)
    assert predictions.logits.shape == (2,)
    assert torch.equal(predictions.logits, direct_logits)


def test_different_sized_graph_requires_explicit_node_features() -> None:
    model, _, _, _, _ = _model_inputs()
    inference_adjacency = build_directed_mean_adjacency(torch.tensor([[0, 4], [4, 1]]), num_nodes=5)

    with pytest.raises(GraphSAGEModelError, match="learned node inputs"):
        model.encode_nodes(inference_adjacency)


def test_config_and_edge_contract_fail_fast() -> None:
    with pytest.raises(GraphSAGEModelError, match="dropout"):
        GraphSAGEConfig(num_nodes=3, transaction_feature_dim=2, dropout=1.0)
    with pytest.raises(GraphSAGEModelError, match="Unknown"):
        GraphSAGEConfig.from_dict(
            {"num_nodes": 3, "transaction_feature_dim": 2, "node_label_dim": 1}
        )

    model, adjacency, senders, receivers, features = _model_inputs()
    with pytest.raises(GraphSAGEModelError, match="torch.long"):
        model(adjacency, senders.float(), receivers, features)
    with pytest.raises(GraphSAGEModelError, match="transaction_features"):
        model(adjacency, senders, receivers, features[:, :2])


def test_deterministic_cpu_helper_sets_seed_and_reports_controls() -> None:
    metadata = configure_deterministic_cpu(44, num_threads=1)
    first_torch = torch.rand(2)
    first_python = random.random()
    configure_deterministic_cpu(44, num_threads=1)

    assert torch.equal(first_torch, torch.rand(2))
    assert first_python == random.random()
    assert metadata["device"] == "cpu"
    assert metadata["deterministic_algorithms"] is True
    assert metadata["num_threads"] == 1
