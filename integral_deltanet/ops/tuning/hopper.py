"""Hopper: GH200 (sm 9.0) launch tables.

Tuned on GH200 at H=16, K=V=128.  H100 shares sm 9.0 and therefore these
tables.  The shared-memory limit (227 KB per block) rules out the C64 readout
state kernel, so the C64 entries below are B300's.
"""

# Readout transport (general_block_triton): forward and backward
# (BK, warps, stages).  The intra and write-solve kernels keep the generic
# launches.
TRANSPORT = (
    {32: (16, 1, 1), 64: (64, 1, 2)},
    {32: (64, 2, 2), 64: (32, 1, 3)},
)

# Readout two-state scan (scan_triton): state forward, state backward, leaf
# backward, causal backward, as in blackwell.INTEGRAL_SCAN.
INTEGRAL_SCAN = (
    {16: (32, 4, 4), 32: (32, 4, 2), 64: (32, 4, 3)},
    {16: (32, 4, 3), 32: (32, 4, 2), 64: (64, 8, 2)},
    {16: (128, 32, 8, 3), 32: (128, 32, 4, 3), 64: (64, 128, 8, 3)},
    {16: (128, 2, 1), 32: (64, 1, 2), 64: (64, 4, 3)},
)
# At B=1 the BV=32 state tiles fill under half of the 132 SMs.  BV=16 doubles
# the programs, which helps there but not once the grid is full, since every
# value tile reloads the C x K operands.  Taken when the pinned state grid
# would occupy at most half of the SMs.
INTEGRAL_SCAN_SMALL_GRID = (
    {16: (16, 4, 4), 32: (16, 4, 3)},
    {16: (16, 4, 2), 32: (16, 4, 2)},
)
# Once a BV=64 state-forward grid covers about the whole GPU (B>=4 at H16),
# halving the value tiles' reloads of the C x K operands beats the extra
# programs.  Taken when that grid fills at least 3/4 of the SMs;
# INTEGRAL_SCAN_LARGE_GRID=0 disables it.
INTEGRAL_SCAN_LARGE_GRID = ({32: (64, 8, 3)},)

# Readout row kernels (readout_triton): (BK, num_warps).
ROWS = {"fwd": (64, 1), "bwd": (32, 1)}
# Long-sequence reload backward (readout_triton): (BK, num_warps).
RELOAD_BWD = (128, 4)
# The reload backward builds its decays as running products here;
# INTEGRAL_READOUT_BWD_PRODUCT=0/1 forces either kernel.
RELOAD_BWD_PRODUCT = True
