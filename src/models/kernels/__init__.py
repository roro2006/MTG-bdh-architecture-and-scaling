"""Custom Pallas kernels for the set encoders and the two interaction arms.

Both arms have a fused kernel here, deliberately. The BDH kernel is a
*validity* fix -- without it the memory traffic and arithmetic that BDH's
accounting assumes are never actually avoided, so the iso-FLOP axis of the
scaling grid would be describing a machine nobody ran. The attention kernel
is a *speed* fix -- attention's analytic FLOP count is already honest. They
are not the same kind of need, but shipping only one of them would mean any
wall-clock comparison between the arms measured kernel effort rather than
architecture (docs/ARCHITECTURE.md, "A fairness note").

Every kernel here ships with a pure-JAX reference and a test asserting the
two agree on values *and* on every gradient. The attention arm is the
control for the whole project; a silently wrong kernel in the control would
invalidate the grid, and a parameter-count test would not catch it.

The set-encoder kernel is a third kind of need again. It is neither a
validity fix nor a marginal speed fix: the encoders are 63-74% of forward
FLOPs and ran entirely on `nn.MultiHeadDotProductAttention`, so without it
the project's "attention-shaped operations are hand-written" commitment
described a minority of the arithmetic (docs/PROJECT_PLAN.md section 5).

Use them through `SetEncoder(fused=True)`, `CrossAttentionArm(fused=True)`
and `BDHArm(fused=True)`, or `ModelConfig(fused_kernels=True)` for all
three at once, rather than by reaching in here --
the fused blocks carry parameter trees identical to the reference blocks,
so the switch is invisible to everything downstream.

Backend note
------------
Pallas lowers to Triton on CUDA and Mosaic on TPU. On a CPU-only install
(no `jaxlib.mlir._mlir_libs._triton_ext`) it can still run under
`interpret=True`, which executes the kernel semantics in pure JAX -- correct
but with none of the fusion benefit, and slower than just calling the
reference. `default_interpret()` picks automatically, so the same code runs
everywhere and the tests pass on a laptop. Never leave interpret mode on
for a real training run.
"""

from .bdh import (
    FusedBDHBlock,
    bdh_gate_decode,
    bdh_scores,
    reference_gate_decode,
    reference_scores,
)
from .cross_attention import (
    FusedCrossAttentionBlock,
    default_interpret,
    fused_attention,
    reference_attention,
)
from .set_encoder import (
    FusedSetAttentionBlock,
    benchmark_set_attention,
    fused_set_attention,
    reference_set_attention,
)

__all__ = [
    "FusedBDHBlock",
    "FusedCrossAttentionBlock",
    "FusedSetAttentionBlock",
    "bdh_gate_decode",
    "bdh_scores",
    "benchmark_set_attention",
    "default_interpret",
    "fused_attention",
    "fused_set_attention",
    "reference_attention",
    "reference_gate_decode",
    "reference_scores",
    "reference_set_attention",
]
