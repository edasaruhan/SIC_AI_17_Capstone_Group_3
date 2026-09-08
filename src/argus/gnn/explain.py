"""Local, model-native input attributions for GraphSAGE transaction scores."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from argus.gnn.adjacency import DirectedMeanAdjacency
from argus.gnn.model import GraphSAGEEdgeClassifier, GraphSAGEModelError


@dataclass(frozen=True)
class EdgeInputAttributions:
    """Per-edge gradients and gradient-times-input contributions.

    Sender and receiver fields explain the classifier's two role-specific
    encoded node inputs. Transaction fields explain supplied transaction
    features. These are local sensitivities, not causal effects.
    """

    logits: Tensor
    probabilities: Tensor
    sender_gradients: Tensor
    receiver_gradients: Tensor
    transaction_gradients: Tensor
    sender_gradient_x_input: Tensor
    receiver_gradient_x_input: Tensor
    transaction_gradient_x_input: Tensor


def explain_edge_inputs(
    model: GraphSAGEEdgeClassifier,
    adjacency: DirectedMeanAdjacency,
    sender_indices: Tensor,
    receiver_indices: Tensor,
    transaction_features: Tensor,
    *,
    node_features: Tensor | None = None,
    target_class: int = 1,
) -> EdgeInputAttributions:
    """Explain each edge target-class logit with gradient-times-input.

    Encoded sender and receiver vectors are detached before differentiation so
    role contributions stay separate, including for self-loop transactions.
    The helper explains classifier inputs rather than historical graph edges.
    """

    if not isinstance(model, GraphSAGEEdgeClassifier):
        raise GraphSAGEModelError("model must be a GraphSAGEEdgeClassifier")
    if target_class not in (0, 1):
        raise GraphSAGEModelError("target_class must be 0 or 1")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            node_embeddings = model.encode_nodes(adjacency, node_features=node_features)
            # Validate before creating role-specific gradient leaves.
            model.classify_edges(
                node_embeddings, sender_indices, receiver_indices, transaction_features
            )
        sender_inputs = (
            node_embeddings.index_select(0, sender_indices).detach().clone().requires_grad_(True)
        )
        receiver_inputs = (
            node_embeddings.index_select(0, receiver_indices).detach().clone().requires_grad_(True)
        )
        transaction_inputs = transaction_features.detach().clone().requires_grad_(True)
        classifier_inputs = torch.cat((sender_inputs, receiver_inputs, transaction_inputs), dim=1)
        logits = model.edge_classifier(classifier_inputs).squeeze(1)
        target_logits = logits if target_class == 1 else -logits
        sender_gradients, receiver_gradients, transaction_gradients = torch.autograd.grad(
            target_logits.sum(),
            (sender_inputs, receiver_inputs, transaction_inputs),
            create_graph=False,
            retain_graph=False,
        )
        return EdgeInputAttributions(
            logits=logits.detach(),
            probabilities=torch.sigmoid(logits.detach()),
            sender_gradients=sender_gradients.detach(),
            receiver_gradients=receiver_gradients.detach(),
            transaction_gradients=transaction_gradients.detach(),
            sender_gradient_x_input=(sender_gradients * sender_inputs).detach(),
            receiver_gradient_x_input=(receiver_gradients * receiver_inputs).detach(),
            transaction_gradient_x_input=(transaction_gradients * transaction_inputs).detach(),
        )
    finally:
        model.train(was_training)
