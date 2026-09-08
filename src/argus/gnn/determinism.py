"""Deterministic CPU controls for reproducible GraphSAGE experiments."""

from __future__ import annotations

import random
from typing import Any

import torch


def configure_deterministic_cpu(seed: int, *, num_threads: int = 1) -> dict[str, Any]:
    """Configure deterministic PyTorch CPU execution and return audit metadata."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads <= 0:
        raise ValueError("num_threads must be a positive integer")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)
    return {
        "seed": seed,
        "device": "cpu",
        "num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_version": torch.__version__,
    }
