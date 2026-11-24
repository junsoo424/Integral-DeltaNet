"""Per-GPU launch tables, grouped by GPU family.

As FLA keys its kernels' launch choices on the platform, every pinned tile
here belongs to the GPU it was tuned on:

    A  ampere     A6000  sm 8.6   ampere/a6000.py
    H  hopper     GH200  sm 9.0   hopper.py
    B  blackwell  B300   sm 10.3  blackwell.py

A family is a module, or a package with a module per GPU (ampere).
The kernels are the same on every GPU.  A kernel module asks for a table by
name and gets ``{capability: table}`` over the GPUs that define it; a GPU
without an entry keeps the generic or autotuned launch, which computes the
same function.  Which fast paths (rather than tiles) a GPU runs is decided in
``ops/arch.py``.
"""
from __future__ import annotations

from . import ampere, blackwell, hopper

ARCHS = {
    (8, 6): ampere.a6000,
    (9, 0): hopper,
    (10, 3): blackwell,
}


def by_arch(name: str) -> dict:
    """``{capability: table}`` for every GPU that pins ``name``."""
    return {
        capability: getattr(module, name)
        for capability, module in ARCHS.items()
        if hasattr(module, name)
    }


__all__ = ["ARCHS", "ampere", "blackwell", "by_arch", "hopper"]
