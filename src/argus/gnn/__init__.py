"""Pure-PyTorch directed GraphSAGE components for transaction classification."""

from argus.gnn.adjacency import (
    DirectedMeanAdjacency,
    GraphAdjacencyError,
    build_directed_mean_adjacency,
)
from argus.gnn.determinism import configure_deterministic_cpu
from argus.gnn.explain import EdgeInputAttributions, explain_edge_inputs
from argus.gnn.model import (
    DirectedGraphSAGEEncoder,
    DirectedGraphSAGELayer,
    EdgePredictions,
    GraphSAGEConfig,
    GraphSAGEEdgeClassifier,
    GraphSAGEModelError,
    predict_edge_scores,
)

__all__ = [
    "DirectedGraphSAGEEncoder",
    "DirectedGraphSAGELayer",
    "DirectedMeanAdjacency",
    "EdgeInputAttributions",
    "EdgePredictions",
    "GraphAdjacencyError",
    "GraphSAGEConfig",
    "GraphSAGEEdgeClassifier",
    "GraphSAGEModelError",
    "build_directed_mean_adjacency",
    "configure_deterministic_cpu",
    "explain_edge_inputs",
    "predict_edge_scores",
]
