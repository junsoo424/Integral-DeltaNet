"""Numerically stable unit-lower triangular solve used by the chunk kernel."""
from __future__ import annotations

import torch


_BLOCK = 16
_FLA_SOLVE_TRIL = None


def _neumann_inverse(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[-1]
    eye = torch.eye(size, dtype=matrix.dtype, device=matrix.device)
    negative_lower = eye - matrix
    inverse = eye + negative_lower
    power = negative_lower @ negative_lower
    covered = 2
    while covered < size:
        inverse = inverse + inverse @ power
        power = power @ power
        covered *= 2
    return inverse


def _block_inverse(matrix: torch.Tensor) -> torch.Tensor:
    """Invert a unit-lower matrix without materializing unstable high powers."""
    size = matrix.shape[-1]
    if size <= _BLOCK:
        return _neumann_inverse(matrix)
    split = size // 2
    top = matrix[..., :split, :split]
    lower = matrix[..., split:, :split]
    bottom = matrix[..., split:, split:]
    top_inverse = _block_inverse(top)
    bottom_inverse = _block_inverse(bottom)
    off_diagonal = -(bottom_inverse @ (lower @ top_inverse))
    zero = torch.zeros_like(top_inverse)
    return torch.cat(
        (
            torch.cat((top_inverse, zero), dim=-1),
            torch.cat((off_diagonal, bottom_inverse), dim=-1),
        ),
        dim=-2,
    )


def _fla_solver():
    global _FLA_SOLVE_TRIL
    if _FLA_SOLVE_TRIL is None:
        try:
            from fla.ops.utils.solve_tril import solve_tril

            _FLA_SOLVE_TRIL = solve_tril
        except Exception:  # pragma: no cover - exercised only without FLA
            _FLA_SOLVE_TRIL = False
    return _FLA_SOLVE_TRIL


def _inverse(
    matrix: torch.Tensor,
    matrix_is_strict_lower: bool = False,
) -> torch.Tensor:
    size = matrix.shape[-1]
    solver = _fla_solver()
    if (
        solver
        and matrix.is_cuda
        and matrix.dtype == torch.float32
        and size in (16, 32, 64)
    ):
        flat = matrix.reshape(-1, size, size)
        if not matrix_is_strict_lower:
            flat = torch.tril(flat, diagonal=-1)
        strict_lower = flat.reshape(flat.shape[0], size, 1, size)
        # Match FLA's production path: the inverse is consumed only by BF16
        # tensor-core GEMMs, so write it to HBM in its consumer dtype instead
        # of materializing FP32 and launching an immediate conversion kernel.
        return solver(
            strict_lower.contiguous(),
            output_dtype=torch.bfloat16,
        ).reshape(matrix.shape)
    if matrix_is_strict_lower:
        matrix = matrix + torch.eye(size, dtype=matrix.dtype, device=matrix.device)
    return _block_inverse(matrix)


class _UnitLowerSolve(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        matrix: torch.Tensor,
        rhs: torch.Tensor,
        matrix_is_strict_lower: bool,
    ) -> torch.Tensor:
        with torch.no_grad():
            inverse = _inverse(matrix, matrix_is_strict_lower)
            if matrix.is_cuda and matrix.dtype == torch.float32:
                inverse_for_apply = inverse.to(torch.bfloat16)
                solution = inverse_for_apply @ rhs.to(torch.bfloat16)
            else:
                inverse_for_apply = inverse
                solution = inverse @ rhs
        ctx.save_for_backward(inverse_for_apply, solution)
        ctx.input_dtypes = (matrix.dtype, rhs.dtype)
        return solution

    @staticmethod
    def backward(ctx, grad_solution: torch.Tensor):
        inverse, solution = ctx.saved_tensors
        matrix_dtype, rhs_dtype = ctx.input_dtypes
        grad_solution = grad_solution.to(inverse.dtype)
        grad_rhs = inverse.transpose(-1, -2) @ grad_solution
        grad_matrix = -(grad_rhs @ solution.transpose(-1, -2))
        return grad_matrix.to(matrix_dtype), grad_rhs.to(rhs_dtype), None


@torch.compiler.disable
def solve_unit_lower(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    matrix_is_strict_lower: bool = False,
) -> torch.Tensor:
    if not matrix.is_cuda and not matrix_is_strict_lower:
        return torch.linalg.solve_triangular(
            matrix, rhs, upper=False, unitriangular=True
        )
    return _UnitLowerSolve.apply(matrix, rhs, matrix_is_strict_lower)


__all__ = ["solve_unit_lower"]
