"""RTX A6000 (sm 8.6) launch tables.

Tuned at H=16, K=V=128; KDA and GDN-2 share every entry, and one entry
serves every batch size and length unless a table says otherwise.  There is
no GENERAL_SCAN table; that scan keeps its autotuned launch.
"""

# Readout transport (general_block_triton): forward and backward
# (BK, warps, stages).
TRANSPORT = (
    {32: (32, 1, 2), 64: (32, 4, 3)},
    {32: (64, 4, 2), 64: (64, 2, 3)},
)

# Direct C32/C64 intra kernels (general_block_triton): forward (BK, warps,
# stages); backward per target block.  These match what the autotuner finds
# and are pinned so the pick does not depend on the shape that happened to
# autotune first.
INTRA = (
    {32: (128, 1, 3), 64: (64, 4, 1)},
    {
        32: ((32, 1, 3), (128, 8, 1)),
        64: ((32, 1, 1), (128, 2, 2), (128, 8, 1), (128, 8, 1)),
    },
)

# Fused value/write solve (general_fused_write_triton): (BK, BV, num_warps,
# num_stages).  As with intra, these equal the autotuner's own pick and pin it.
FUSED_WRITE_SOLVE = (
    {32: (32, 128, 1, 1), 64: (32, 32, 1, 1)},
    {32: (32, 64, 8, 2), 64: (64, 128, 8, 1)},
)

# Readout two-state scan (scan_triton): state forward, state backward, leaf
# backward, causal backward, as in blackwell.INTEGRAL_SCAN.  Without it every
# one of these kernels autotunes, and fla caches the pick on disk keyed
# without T.  The state scans hold one program per (sequence, value tile) for
# the whole length, and the best tile follows batch*heads: with many sequences
# several programs share an SM and one stage (less shared memory, more
# residents) is faster; with few, every program has an SM to itself and a
# second stage hides the operand loads instead.  Such entries are
# ((min_batch_heads, tile), ...), largest threshold first.
INTEGRAL_SCAN = (
    {32: ((32, (32, 4, 1)), (0, (32, 4, 2))),
     64: ((32, (32, 4, 1)), (0, (32, 8, 2)))},
    {32: ((64, (64, 4, 1)), (32, (64, 8, 2)), (0, (32, 8, 2))),
     64: ((32, (64, 8, 1)), (0, (32, 8, 2)))},
    {32: (128, 64, 8, 1), 64: (128, 32, 8, 3)},
    {32: (64, 2, 1), 64: (64, 4, 2)},
)

# Long-sequence reload backward (readout_triton): (BK, num_warps).  At K=128
# the CP8 reload backward spills far fewer registers than CP4.  The row kernels
# keep the generic ROWS_CONFIG.
RELOAD_BWD = (128, 4)
