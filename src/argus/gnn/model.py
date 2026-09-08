"""Pure-PyTorch directed GraphSAGE transaction classifier."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from argus.gnn.adjacency import DirectedMeanAdjacency


class GraphSAGEModelError(ValueError):
    """Raised when GraphSAGE configuration or edge inputs are invalid."""


def _positive_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GraphSAGEModelError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class GraphSAGEConfig:
    """JSON-friendly architecture configuration for the edge classifier."""

    num_nodes: int
    transaction_feature_dim: int
    node_input_dim: int = 32
    hidden_dim: int = 64
    embedding_dim: int = 32
    num_layers: int = 2
    classifier_hidden_dim: int = 64
    dropout: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "num_nodes",
            "transaction_feature_dim",
            "node_input_dim",
            "hidden_dim",
            "embedding_dim",
            "num_layers",
            "classifier_hidden_dim",
        ):
            _positive_integer(getattr(self, name), name=name)
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise GraphSAGEModelError("dropout must be a number in [0, 1)")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise GraphSAGEModelError("dropout must be a number in [0, 1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise GraphSAGEModelError("seed must be a non-negative integer")

    def to_dict(self) -> dict[str, int | float]:
        """Return the configuration as a JSON-serialisable mapping."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GraphSAGEConfig:
        """Build a validated config while rejecting misspelled fields."""

        if not isinstance(payload, Mapping):
            raise GraphSAGEModelError("GraphSAGE config must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(payload).difference(allowed))
        if unknown:
            raise GraphSAGEModelError(f"Unknown GraphSAGE config fields: {unknown}")
        try:
            return cls(**dict(payload))
        except TypeError as error:
            raise GraphSAGEModelError(f"Invalid GraphSAGE config: {error}") from error


class DirectedGraphSAGELayer(nn.Module):
    """Combine self, inbound-neighbour mean, and outbound-neighbour mean."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        _positive_integer(input_dim, name="input_dim")
        _positive_integer(output_dim, name="output_dim")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(3 * input_dim, output_dim)

    def forward(self, node_states: Tensor, adjacency: DirectedMeanAdjacency) -> Tensor:
        if (
            not isinstance(node_states, Tensor)
            or node_states.ndim != 2
            or node_states.shape[1] != self.input_dim
        ):
            raise GraphSAGEModelError(f"node_states must have shape (n_nodes, {self.input_dim})")
        inbound_mean, outbound_mean = adjacency.aggregate(node_states)
        return self.linear(torch.cat((node_states, inbound_mean, outbound_mean), dim=1))


class DirectedGraphSAGEEncoder(nn.Module):
    """Stack directed mean-aggregation GraphSAGE layers."""

    def __init__(self, config: GraphSAGEConfig) -> None:
        super().__init__()
        dimensions = [config.node_input_dim]
        dimensions.extend([config.hidden_dim] * max(config.num_layers - 1, 0))
        dimensions.append(config.embedding_dim)
        self.layers = nn.ModuleList(
            DirectedGraphSAGELayer(input_dim, output_dim)
            for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:], strict=True)
        )
        self.dropout = float(config.dropout)

    def forward(self, node_states: Tensor, adjacency: DirectedMeanAdjacency) -> Tensor:
        output = node_states
        for index, layer in enumerate(self.layers):
            output = layer(output, adjacency)
            if index < len(self.layers) - 1:
                output = F.relu(output)
                output = F.dropout(output, p=self.dropout, training=self.training)
        return output


@dataclass(frozen=True)
class EdgePredictions:
    """Binary transaction scores on raw-logit and probability scales."""

    logits: Tensor
    probabilities: Tensor


class GraphSAGEEdgeClassifier(nn.Module):
    """Classify edges from sender, receiver, and transaction features.

    The supervised target belongs only to transactions. No account/node fraud
    labels are accepted or created by this model.
    """

    checkpoint_format = "argus.graphsage_edge_classifier.v1"

    def __init__(self, config: GraphSAGEConfig) -> None:
        super().__init__()
        if not isinstance(config, GraphSAGEConfig):
            raise GraphSAGEModelError("config must be a GraphSAGEConfig")
        self.config = config
        # Reproducible architecture initialisation without consuming the caller's
        # global PyTorch random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.node_inputs = nn.Embedding(config.num_nodes, config.node_input_dim)
            self.encoder = DirectedGraphSAGEEncoder(config)
            classifier_input_dim = 2 * config.embedding_dim + config.transaction_feature_dim
            self.edge_classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, config.classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(config.classifier_hidden_dim, 1),
            )

    @property
    def parameter_device(self) -> torch.device:
        """Return the device currently holding model parameters."""

        return self.node_inputs.weight.device

    @property
    def parameter_dtype(self) -> torch.dtype:
        """Return the floating-point dtype used by model parameters."""

        return self.node_inputs.weight.dtype

    def encode_nodes(
        self,
        adjacency: DirectedMeanAdjacency,
        *,
        node_features: Tensor | None = None,
    ) -> Tensor:
        """Encode all local nodes using learned or supplied initial inputs."""

        if not isinstance(adjacency, DirectedMeanAdjacency):
            raise GraphSAGEModelError("adjacency must be a DirectedMeanAdjacency")
        if adjacency.device != self.parameter_device or adjacency.dtype != self.parameter_dtype:
            raise GraphSAGEModelError(
                "adjacency must share the model parameter device and dtype "
                f"({self.parameter_device}, {self.parameter_dtype})"
            )
        if node_features is None:
            if adjacency.num_nodes != self.config.num_nodes:
                raise GraphSAGEModelError(
                    "learned node inputs require an adjacency with "
                    f"{self.config.num_nodes} nodes; got {adjacency.num_nodes}"
                )
            initial = self.node_inputs.weight
        else:
            expected = (adjacency.num_nodes, self.config.node_input_dim)
            if not isinstance(node_features, Tensor) or tuple(node_features.shape) != expected:
                raise GraphSAGEModelError(
                    "node_features must have shape "
                    f"{expected}; got {getattr(node_features, 'shape', None)}"
                )
            if (
                node_features.device != self.parameter_device
                or node_features.dtype != self.parameter_dtype
            ):
                raise GraphSAGEModelError(
                    "node_features must share the model parameter device and dtype"
                )
            initial = node_features
        return self.encoder(initial, adjacency)

    def _validate_edge_inputs(
        self,
        node_embeddings: Tensor,
        sender_indices: Tensor,
        receiver_indices: Tensor,
        transaction_features: Tensor,
    ) -> None:
        if (
            not isinstance(node_embeddings, Tensor)
            or node_embeddings.ndim != 2
            or node_embeddings.shape[0] <= 0
            or node_embeddings.shape[1] != self.config.embedding_dim
        ):
            raise GraphSAGEModelError(
                "node_embeddings must have shape "
                f"(n_nodes, {self.config.embedding_dim}) with n_nodes > 0; "
                f"got {getattr(node_embeddings, 'shape', None)}"
            )
        node_count = node_embeddings.shape[0]
        for name, indices in (
            ("sender_indices", sender_indices),
            ("receiver_indices", receiver_indices),
        ):
            if not isinstance(indices, Tensor) or indices.ndim != 1 or indices.dtype != torch.long:
                raise GraphSAGEModelError(f"{name} must be a rank-one torch.long tensor")
            if indices.device != self.parameter_device:
                raise GraphSAGEModelError(f"{name} must be on {self.parameter_device}")
            if indices.numel() and (
                int(indices.min().item()) < 0 or int(indices.max().item()) >= node_count
            ):
                raise GraphSAGEModelError(f"{name} values must lie in [0, {node_count})")
        edge_count = sender_indices.numel()
        if receiver_indices.numel() != edge_count:
            raise GraphSAGEModelError("sender and receiver index counts must match")
        expected_features = (edge_count, self.config.transaction_feature_dim)
        if (
            not isinstance(transaction_features, Tensor)
            or tuple(transaction_features.shape) != expected_features
        ):
            raise GraphSAGEModelError(
                "transaction_features must have shape "
                f"{expected_features}; got {getattr(transaction_features, 'shape', None)}"
            )
        if (
            node_embeddings.device != self.parameter_device
            or transaction_features.device != self.parameter_device
        ):
            raise GraphSAGEModelError("edge tensors must be on the model parameter device")
        if (
            node_embeddings.dtype != self.parameter_dtype
            or transaction_features.dtype != self.parameter_dtype
        ):
            raise GraphSAGEModelError("edge feature tensors must use the model parameter dtype")

    def classify_edges(
        self,
        node_embeddings: Tensor,
        sender_indices: Tensor,
        receiver_indices: Tensor,
        transaction_features: Tensor,
    ) -> Tensor:
        """Return one raw binary logit for each transaction edge."""

        self._validate_edge_inputs(
            node_embeddings, sender_indices, receiver_indices, transaction_features
        )
        sender_embeddings = node_embeddings.index_select(0, sender_indices)
        receiver_embeddings = node_embeddings.index_select(0, receiver_indices)
        classifier_inputs = torch.cat(
            (sender_embeddings, receiver_embeddings, transaction_features), dim=1
        )
        return self.edge_classifier(classifier_inputs).squeeze(1)

    def forward(
        self,
        adjacency: DirectedMeanAdjacency,
        sender_indices: Tensor,
        receiver_indices: Tensor,
        transaction_features: Tensor,
        *,
        node_features: Tensor | None = None,
    ) -> Tensor:
        """Encode the graph and return raw transaction logits."""

        node_embeddings = self.encode_nodes(adjacency, node_features=node_features)
        return self.classify_edges(
            node_embeddings, sender_indices, receiver_indices, transaction_features
        )

    def export_checkpoint(self) -> dict[str, Any]:
        """Return model config and CPU weights suitable for ``torch.save``."""

        return {
            "format": self.checkpoint_format,
            "config": self.config.to_dict(),
            "state_dict": {
                name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        map_location: torch.device | str = "cpu",
    ) -> GraphSAGEEdgeClassifier:
        """Restore a model from :meth:`export_checkpoint` output."""

        if not isinstance(checkpoint, Mapping):
            raise GraphSAGEModelError("GraphSAGE checkpoint must be a mapping")
        if checkpoint.get("format") != cls.checkpoint_format:
            raise GraphSAGEModelError("Unsupported GraphSAGE checkpoint format")
        config_payload = checkpoint.get("config")
        state = checkpoint.get("state_dict")
        if not isinstance(config_payload, Mapping) or not isinstance(state, Mapping):
            raise GraphSAGEModelError("GraphSAGE checkpoint is missing config or state_dict")
        model = cls(GraphSAGEConfig.from_dict(config_payload)).to(map_location)
        try:
            model.load_state_dict(dict(state), strict=True)
        except (RuntimeError, TypeError) as error:
            raise GraphSAGEModelError(f"Invalid GraphSAGE state_dict: {error}") from error
        return model


def predict_edge_scores(
    model: GraphSAGEEdgeClassifier,
    adjacency: DirectedMeanAdjacency,
    sender_indices: Tensor,
    receiver_indices: Tensor,
    transaction_features: Tensor,
    *,
    node_features: Tensor | None = None,
    batch_size: int | None = None,
) -> EdgePredictions:
    """Predict raw logits and sigmoid probabilities without retaining gradients."""

    if not isinstance(model, GraphSAGEEdgeClassifier):
        raise GraphSAGEModelError("model must be a GraphSAGEEdgeClassifier")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            node_embeddings = model.encode_nodes(adjacency, node_features=node_features)
            model._validate_edge_inputs(
                node_embeddings, sender_indices, receiver_indices, transaction_features
            )
            edge_count = sender_indices.numel()
            if batch_size is None:
                batch_size = max(edge_count, 1)
            _positive_integer(batch_size, name="batch_size")
            batches: list[Tensor] = []
            for start in range(0, edge_count, batch_size):
                stop = min(start + batch_size, edge_count)
                batches.append(
                    model.classify_edges(
                        node_embeddings,
                        sender_indices[start:stop],
                        receiver_indices[start:stop],
                        transaction_features[start:stop],
                    )
                )
            if batches:
                logits = torch.cat(batches)
            else:
                logits = model.classify_edges(
                    node_embeddings,
                    sender_indices,
                    receiver_indices,
                    transaction_features,
                )
            return EdgePredictions(logits=logits, probabilities=torch.sigmoid(logits))
    finally:
        model.train(was_training)
