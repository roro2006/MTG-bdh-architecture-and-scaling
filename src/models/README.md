# Models

Two interaction arms sharing one front end (see `docs/ARCHITECTURE.md`). Everything except the arm is identical between them, so whatever difference shows up is attributable to the mechanism and nothing upstream of it.

| file | role | status |
|---|---|---|
| `embeddings.py` | composite card embeddings + pack/pick context features | done |
| `set_encoder.py` | permutation-invariant pack and pool encoders | done (flax attention, or the Pallas kernel under `fused=True`) |
| `attention_arm.py` | pool-to-pack cross-attention | done |
| `bdh_arm.py` | JAX port of BDH's sparse/Hebbian block | done |
| `kernels/` | hand-written Pallas kernels | both arms and the set encoder done; none benchmarked on hardware |
| `pick_model.py` | assembly, pointer head, analytic parameter counts | done |

## Card embeddings are composed, not looked up

`CardEmbedding` projects the attribute block built by `src/data/card_features.py` — colour identity, castable colours, type flags, mana value, rarity, power/toughness, a fixed global keyword vocabulary, and structured mechanical features derived from oracle text — rather than indexing a table keyed by card id. A card printed after training gets a usable vector from its attributes alone, which is the property the whole zero-shot goal depends on.

The module reads the feature width from the table (`card_feature_dim=table.shape[1]`), so widening the representation needs no change here.

## Where synergy lives

Not in the embeddings. Synergy is a relation between two cards, and a per-card vector has nowhere to put a relation. The feature table's job is to make the relation representable — to carry what each card mechanically does — and the **interaction arm** is what computes it, as a learned bilinear form between a candidate (query) and the pool (context).

This is why `src/analysis/` exists: top-1 accuracy cannot distinguish a model that learned card interaction from one that learned colour-matching, so the pool-ablation and fixed-pack/varying-pool probes are the actual evidence.

## Properties that are structural, not learned

Asserted in the architecture doc and enforced in `tests/test_models.py`:

- **Pack order permutes the logits and changes nothing else**, and **pool order changes nothing at all.** No positional encoding exists anywhere in the front end.
- **The output space is closed to the pack.** The pointer head pins padding slots to `MASK_SCORE`, so softmax puts exactly zero mass outside the pack and the model cannot express an impossible pick.
- **A one-card pack has loss exactly zero.** There is no decision at the last pick of a pack.
- **An empty pool does not produce NaNs.** At pack 0 pick 0 the pool is empty, so every cross-attention key would be masked and the softmax would divide by zero — 140,237 rows of the FIN corpus, one in every 42. `CrossAttentionArm` carries a learned null key that is always visible, which is both the numerically safe fix and the semantically right one: "I have no cards yet" is a real state. **BDH needs no such fix**: it has no softmax, so no denominator, and an empty pool simply produces a zero score. Tested either way.

## The two arms

`BDHArm` takes the same arguments in the same order as `CrossAttentionArm` and returns the same shape, so `PickModel` swaps one for the other and changes nothing else. Two deviations from the BDH reference were forced by commitments made elsewhere, and both are argued at length in `bdh_arm.py`: **RoPE and the causal mask are dropped** (both encode order, and a pool has none).

**Sizing is no longer constrained.** `neuron_multiplier` defaulted to 4 to make an iso-parameter comparison possible against cross-attention — the reference's own 128 is 32× the attention arm's parameter count. With BDH now judged on how well it drafts rather than on a scaling-exponent comparison, that constraint is lifted and larger neuron widths are available. This matters for the kernel: at `neuron_multiplier=4` and `d=64` the per-head neuron width is 64, which is a single tile at the default `BLOCK_N`, so the neuron-axis tiling the BDH kernel is built around does nothing. It starts doing something at 8 tiles (multiplier 32) and does the thing it was designed for at 32 tiles (multiplier 128).

## Kernels

`kernels/` holds hand-written Pallas implementations, selected with `ModelConfig(fused_kernels=True)`, `Arm(fused=True)`, or `--fused-kernels`. Parameter trees are identical to the reference blocks name for name, so the switch is invisible to checkpointing, parameter counting and everything downstream.

**Scope:** attention-shaped and neuron-space operations are hand-written; LayerNorm, Dense, softmax and elementwise ops stay in XLA, which fuses them competently and whose gradients are not worth re-deriving to save nothing.

The two arms solve opposite problems. Cross-attention is small enough that a whole `(batch, head)` slice fits in SRAM — pack ≤ 14, pool ≤ 42 — so the kernel skips FlashAttention's tiling and online softmax entirely and does forward and backward in one block each, keeping only the log-sum-exp so the backward can reconstruct the probabilities. BDH's problem is the reverse: its `(B, nh, L, N)` neuron tensors are the largest things in the model and needed nowhere outside the block, so the neuron axis becomes a sequential grid dimension and never reaches HBM.

**The gap, now closed:** the set encoders are 58% of parameters and 63–74% of forward FLOPs run outside the arm, all of it formerly on `nn.MultiHeadDotProductAttention`. `kernels/set_encoder.py` is the kernel for that masked, position-free self-attention. `fused=True` selects it, leaving the parameter tree and the values untouched, so which path runs is an execution choice and not a model one.

Three things worth knowing before using these:

- **They win memory traffic, not FLOPs.** Unstructured zeros still occupy a tensor-core lane. See the accounting section in `docs/ARCHITECTURE.md`.
- **They are correct but unmeasured.** On a CPU-only box the tests run under Pallas's `interpret=True`, which checks semantics but not lowering and does no fusion at all. Set `KERNEL_INTERPRET=0` on a GPU or TPU to exercise the real thing, and benchmark against the references before claiming anything. This has not happened yet: the T4 runs in `docs/RESULTS.md` were trained with `fused_kernels: false`, so no kernel here has ever been lowered on an accelerator.
- **A fully-masked query row is defined as zero**, not as flax's uniform-softmax-over-everything. Those rows are padded pack slots that `PointerHead` discards; flax's value for them depends on how far the key axis happens to be padded, which would make the same model give different numbers at different block sizes. `EMPTY_ROW_NOTE` in `kernels/cross_attention.py` has the detail.

`tests/test_kernels.py` asserts every kernel against a pure-JAX reference on values *and* on every gradient, and both fused arms against their reference arms under one shared parameter set. That is stricter than it may look necessary: a wrong kernel would not fail visibly, it would produce plausible numbers that happen to be false, and a parameter-count test cannot catch it.

## Parameter counts are derived

`count_params_analytic` builds N term by term from the config — layer norms, attention projections and their biases, MLPs, embeddings — and the test suite asserts it against the realised pytree at four widths. `train_model` re-checks it before every run and refuses to train on a mismatch.

This matters because the model's size is chosen by a fit, not by taste (`docs/PROJECT_PLAN.md` §6), and a sizing decision made on a number nobody verified is a guess wearing a derivation's clothes.

Verified: analytic equals realised at d = 32, 64, 128, 256 and 512 (67,681 → 16,074,241 parameters).
