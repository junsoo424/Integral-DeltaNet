"""Blackwell: B300 (sm 10.3) launch tables -- the setting of the paper's numbers.

Tuned at the B=8, T=4096, H=8, K=V=128 training shape unless a table says
otherwise.  The autotuner's own picks vary between runs, so the training shape
is pinned even where the autotuner is already close.
"""

# Delta-rule solve (delta_triton): (BK, BV, num_warps, num_stages).  The
# C16/C32 recompute forward keeps autotuning.
DELTA_SOLVE = (
    {64: (32, 64, 1, 2)},
    {16: (32, 64, 1, 3), 32: (32, 64, 1, 3), 64: (32, 64, 2, 3)},
)

# Fused GDN builder forward (gdn_fused_triton): (BK, BV, num_warps,
# num_stages).  Both picks give results bitwise identical to the default
# launch.  A BK=32 tile would change the bf16 rounding of the Gram
# accumulation, so it is not used.
BUILDER_FORWARD = {16: (128, 64, 2, 3), 32: (128, 64, 1, 3)}

# Fused value/write solve (general_fused_write_triton): (BK, BV, num_warps,
# num_stages), shared by KDA and GDN-2.
FUSED_WRITE_SOLVE = (
    {32: (32, 64, 1, 3), 64: (32, 32, 1, 2)},
    {32: (32, 64, 1, 3), 64: (32, 64, 4, 3)},
)

# Readout transport (general_block_triton): forward and backward
# (BK, warps, stages).
TRANSPORT = (
    {32: (64, 1, 2), 64: (64, 1, 2)},
    {32: (64, 1, 3), 64: (32, 1, 3)},
)

# Direct C32/C64 intra kernels (general_block_triton), shared by KDA and
# GDN-2: forward (BK, warps, stages); backward entries are per target block,
# as in ampere/a6000.INTRA.
INTRA = (
    {32: (64, 1, 2), 64: (64, 1, 2)},
    {
        32: ((64, 4, 1), (64, 2, 2)),
        64: ((128, 2, 1), (64, 2, 2), (64, 2, 2), (64, 4, 2)),
    },
)

# Non-shared KDA/GDN-2 scan (scan_triton), including BV=64 tiles: state
# forward (BV, warps, stages), state backward, leaf backward (BK, BV, warps,
# stages), causal backward (BV, warps, stages).
GENERAL_SCAN = (
    {32: (32, 4, 2), 64: (64, 8, 2)},
    {32: (32, 4, 2), 64: (64, 8, 2)},
    {32: (64, 32, 4, 3), 64: (64, 64, 4, 2)},
    {32: (64, 4, 3), 64: (64, 4, 3)},
)

# Readout two-state scan (scan_triton) at B8/T4096/H8: the triangular
# recurrence needs different C16 backward and C64 leaf tiles from GENERAL_SCAN.
INTEGRAL_SCAN = (
    {16: (32, 4, 2), 32: (64, 8, 2), 64: (32, 4, 3)},
    {16: (32, 4, 2), 32: (32, 4, 3), 64: (64, 8, 2)},
    {16: (64, 16, 2, 3), 32: (64, 32, 4, 3), 64: (64, 128, 8, 3)},
    {16: (64, 4, 3), 32: (64, 4, 3), 64: (64, 4, 3)},
)

# Shared-gate DeltaNet/GDN scan (scan_triton).  The leaf pass wants tiles
# outside the autotuner's list (BK=32 at C16, BV=16 at C32, BV=128 at C64);
# the state kernels are pinned to remove the autotuner's variance.
SCALAR_SCAN = (
    {16: (32, 4, 2), 32: (64, 8, 2), 64: (64, 8, 2)},
    {16: (32, 4, 2), 32: (64, 8, 2), 64: (32, 8, 3)},
    {16: (32, 32, 2, 3), 32: (64, 16, 2, 3), 64: (64, 128, 8, 2)},
    {16: (128, 1, 2), 32: (128, 2, 1), 64: (64, 4, 1)},
)

# Long-sequence reload backward (readout_triton): (BK, num_warps).
RELOAD_BWD = (128, 4)
