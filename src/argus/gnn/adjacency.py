"""Sparse directed adjacency utilities for GraphSAGE message passing.

The helpers intentionally depend only on PyTorch. One ``edge_index`` column is
one observed transaction, so repeated transfers remain frequency-weighted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


class GraphAdjacencyError(ValueError):
    """Raised when a directed sparse adjacency cannot be constructed safely."""


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GraphAdjacencyError(f"{name} must be a positive integer")
    return value


def _validate_sparse_square(matrix: Tensor, *, num_nodes: int, name: str) -> Tensor:
    if not isinstance(matrix, Tensor) or matrix.layout != torch.sparse_coo:
        raise GraphAdjacencyError(f"{name} must be a sparse COO tensor")
    if matrix.ndim != 2 or tuple(matrix.shape) != (num_nodes, num_nodes):
        raise GraphAdjacencyError(
            f"{name} must have shape ({num_nodes}, {num_nodes}); got {tuple(matrix.shape)}"
        )
    if not matrix.dtype.is_floating_point:
        raise GraphAdjacencyError(f"{name} values must use a floating-point dtype")
    return matrix.coalesce()


@dataclass(frozen=True)
class DirectedMeanAdjacency:
    """Row-normalised sparse matrices for inbound and outbound aggregation.

    ``incoming[v, u]`` is non-zero for transaction edges ``u -> v`` while
    ``outgoing[u, v]`` is non-zero for the same edges. Nodes with no neighbours
    in a direction receive an all-zero aggregate.
    """

    incoming: Tensor
    outgoing: Tensor
    num_nodes: int
    edge_count: int

    def __post_init__(self) -> None:
        num_nodes = _positive_integer(self.num_nodes, name="num_nodes")
        if isinstance(self.edge_count, bool) or not isinstance(self.edge_count, int):
            raise GraphAdjacencyError("edge_count must be a non-negative integer")
        if self.edge_count < 0:
            raise GraphAdjacencyError("edge_count must be a non-negative integer")
        incoming = _validate_sparse_square(self.incoming, num_nodes=num_nodes, name="incoming")
        outgoing = _validate_sparse_square(self.outgoing, num_nodes=num_nodes, name="outgoing")
        if incoming.device != outgoing.device or incoming.dtype != outgoing.dtype:
            raise GraphAdjacencyError("incoming and outgoing matrices must share device and dtype")
        object.__setattr__(self, "incoming", incoming)
        object.__setattr__(self, "outgoing", outgoing)

    @property
    def device(self) -> torch.device:
        """Return the device holding both sparse matrices."""

        return self.incoming.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating-point dtype used by aggregation weights."""

        return self.incoming.dtype

    def aggregate(self, node_states: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(inbound_mean, outbound_mean)`` for every node."""

        if not isinstance(node_states, Tensor) or node_states.ndim != 2:
            raise GraphAdjacencyError("node_states must be a rank-two tensor")
        if node_states.shape[0] != self.num_nodes:
            raise GraphAdjacencyError(
                f"node_states has {node_states.shape[0]} rows; expected {self.num_nodes}"
            )
        if not node_states.dtype.is_floating_point:
            raise GraphAdjacencyError("node_states must use a floating-point dtype")
        if node_states.device != self.device or node_states.dtype != self.dtype:
            raise GraphAdjacencyError(
                "node_states must share the adjacency device and dtype "
                f"({self.device}, {self.dtype})"
            )
        return (
            torch.sparse.mm(self.incoming, node_states),
            torch.sparse.mm(self.outgoing, node_states),
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> DirectedMeanAdjacency:
        """Return an adjacency copied to ``device``/``dtype`` when requested."""

        target_dtype = self.dtype if dtype is None else dtype
        if not isinstance(target_dtype, torch.dtype) or not target_dtype.is_floating_point:
            raise GraphAdjacencyError("adjacency dtype must be floating point")
        return DirectedMeanAdjacency(
            incoming=self.incoming.to(device=device, dtype=target_dtype),
            outgoing=self.outgoing.to(device=device, dtype=target_dtype),
            num_nodes=self.num_nodes,
            edge_count=self.edge_count,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a compact CPU representation suitable for ``torch.save``."""

        incoming = self.incoming.coalesce().cpu()
        outgoing = self.outgoing.coalesce().cpu()
        return {
            "format": "argus.directed_mean_adjacency.v1",
            "num_nodes": self.num_nodes,
            "edge_count": self.edge_count,
            "dtype": str(self.dtype).removeprefix("torch."),
            "incoming_indices": incoming.indices().clone(),
            "incoming_values": incoming.values().clone(),
            "outgoing_indices": outgoing.indices().clone(),
            "outgoing_values": outgoing.values().clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> DirectedMeanAdjacency:
        """Restore an adjacency produced by :meth:`state_dict`."""

        if not isinstance(state, Mapping):
            raise GraphAdjacencyError("Directed adjacency checkpoint must be a mapping")
        if state.get("format") != "argus.directed_mean_adjacency.v1":
            raise GraphAdjacencyError("Unsupported directed adjacency checkpoint format")
        num_nodes = _positive_integer(state.get("num_nodes"), name="num_nodes")
        edge_count = state.get("edge_count")
        if isinstance(edge_count, bool) or not isinstance(edge_count, int) or edge_count < 0:
            raise GraphAdjacencyError("edge_count must be a non-negative integer")
        dtype_name = state.get("dtype")
        dtype = getattr(torch, dtype_name, None) if isinstance(dtype_name, str) else None
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise GraphAdjacencyError("Checkpoint contains an unsupported adjacency dtype")

        def restore(prefix: str) -> Tensor:
            indices = state.get(f"{prefix}_indices")
            values = state.get(f"{prefix}_values")
            if not isinstance(indices, Tensor) or not isinstance(values, Tensor):
                raise GraphAdjacencyError(f"Checkpoint is missing {prefix} sparse tensors")
            return torch.sparse_coo_tensor(
                indices.to(device=device, dtype=torch.long),
                values.to(device=device, dtype=dtype),
                size=(num_nodes, num_nodes),
                device=device,
                dtype=dtype,
                check_invariants=False,
            ).coalesce()

        return cls(
            incoming=restore("incoming"),
            outgoing=restore("outgoing"),
            num_nodes=num_nodes,
            edge_count=edge_count,
        )


def _normalised_sparse_matrix(
    row_indices: Tensor,
    column_indices: Tensor,
    *,
    num_nodes: int,
    dtype: torch.dtype,
) -> Tensor:
    device = row_indices.device
    values = torch.ones(row_indices.numel(), dtype=dtype, device=device)
    counts = torch.sparse_coo_tensor(
        torch.stack((row_indices, column_indices), dim=0),
        values,
        size=(num_nodes, num_nodes),
        dtype=dtype,
        device=device,
        check_invariants=False,
    ).coalesce()
    if counts._nnz() == 0:
        return counts
    indices = counts.indices()
    count_values = counts.values()
    row_totals = torch.zeros(num_nodes, dtype=dtype, device=device)
    row_totals.scatter_add_(0, indices[0], count_values)
    mean_values = count_values / row_totals.index_select(0, indices[0])
    return torch.sparse_coo_tensor(
        indices,
        mean_values,
        size=(num_nodes, num_nodes),
        dtype=dtype,
        device=device,
        check_invariants=False,
    ).coalesce()


def build_directed_mean_adjacency(
    edge_index: Tensor,
    *,
    num_nodes: int,
    dtype: torch.dtype = torch.float32,
) -> DirectedMeanAdjacency:
    """Build directed row-normalised sparse adjacency from ``[source, target]`` edges."""

    num_nodes = _positive_integer(num_nodes, name="num_nodes")
    if not isinstance(edge_index, Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise GraphAdjacencyError("edge_index must be a rank-two tensor with shape (2, n_edges)")
    if edge_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise GraphAdjacencyError("edge_index must use an integer dtype")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise GraphAdjacencyError("dtype must be a floating-point torch dtype")
    edge_index = edge_index.to(dtype=torch.long)
    if edge_index.numel() and (
        int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= num_nodes
    ):
        raise GraphAdjacencyError(f"edge_index values must lie in [0, {num_nodes})")
    source, target = edge_index[0], edge_index[1]
    return DirectedMeanAdjacency(
        incoming=_normalised_sparse_matrix(target, source, num_nodes=num_nodes, dtype=dtype),
        outgoing=_normalised_sparse_matrix(source, target, num_nodes=num_nodes, dtype=dtype),
        num_nodes=num_nodes,
        edge_count=edge_index.shape[1],
    )
