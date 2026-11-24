"""Ampere launch tables, one module per GPU.

    a6000  RTX A6000  sm 8.6

Other Ampere GPUs have no tables and keep the generic or autotuned launches.
"""
from __future__ import annotations

from . import a6000

__all__ = ["a6000"]
