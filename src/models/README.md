# Models

Two architectures sharing one front-end (see `docs/ARCHITECTURE.md`). Everything except the interaction arm is identical between the two sides of the scaling grid, so whatever difference shows up in the curves is attributable to the mechanism and nothing upstream of it.

| file | role | status |
|---|---|---|
| `embeddings.py` | composite card embeddings + pack/pick context features | done |
| `set_encoder.py` | permutation-invariant pack and pool encoders | done |
| `attention_arm.py` | pool-to-pack cross-attention | done |
| `bdh_arm.py` | JAX port of BDH's sparse/Hebbian block | done |
| `kernels/` | fused Pallas kernels for both arms | done |
| `pick_model.py` | assembly, pointer head, analytic parameter counts | done |

## Card embeddings are composed, not looked up

`CardEmbedding` projects the 65-dimensional attribute block built by `src/data/card_features.py` — colour identity, castable colours, type flags, keyword flags, rarity, mana value, power/toughness — rather than indexing a table keyed by card id. A card printed after training gets a usable vector from its attributes alone.

One thing that came out of building this and is worth knowing before touching the feature table: Scryfall's `keywords` field is not just evergreen mechanics. FIN lists flavour-named abilities there too, and **84 of its 118 keywords appear on exactly one card**. A feature column set for exactly one card is that card's id in disguise — it would have handed the model 84 free one-hot identity columns and quietly confounded the N-axis of the scaling study. `MIN_KEYWORD_CARDS` drops them; the comment there explains the trade.

## Properties that are structural, not learned

These are asserted in the architecture doc and enforced in `tests/test_models.py`:

- **Pack order permutes the logits and changes nothing else**, and **pool order changes nothing at all.** No positional encoding exists anywhere in the front end.
- **The output space is closed to the pack.** The pointer head pins padding slots to `MASK_SCORE`, so softmax puts exactly zero mass outside the pack and the model cannot express an impossible pick.
- **A one-card pack has loss exactly zero.** There is no decision at pick 13.
- **An empty pool does not produce NaNs.** At pack 0 pick 0 the pool is empty, so every cross-attention key would be masked and the softmax would divide by zero — 140,237 rows of the FIN corpus, one in every 42. `CrossAttentionArm` carries a learned null key that is always visible, which is both the numerically safe fix and the semantically right one: "I have no cards yet" is a real state. **BDH needs no such fix**: it has no softmax, so no denominator, and an empty pool simply produces a zero score. Tested either way.

## The two arms are the same experiment

`BDHArm` takes the same arguments in the same order as `CrossAttentionArm` and returns the same shape, so `PickModel` swaps one for the other and changes nothing else. Two deviations from the BDH reference were forced by commitments this project had already made, and both are argued at length in `bdh_arm.py`: **RoPE and the causal mask are dropped** (both encode order, and a pool has none), and **`neuron_multiplier` defaults to 4 rather than the reference's 128** (which would be 32× the attention arm's parameter count and make an iso-parameter grid impossible). At 4 the two arms match to 0.5% at d=256.

## Kernels

`kernels/` holds fused Pallas implementations of both arms, selected with `ModelConfig(fused_kernels=True)`, `Arm(fused=True)`, or `--fused-kernels`. Parameter trees are identical to the reference blocks name for name, so the switch is invisible to checkpointing, parameter counting and everything downstream.

They solve opposite problems. Cross-attention is small enough that a whole `(batch, head)` slice fits in SRAM — pack ≤ 14, pool ≤ 42 — so the kernel skips FlashAttention's tiling and online softmax entirely and does forward and backward in one block each, keeping only the log-sum-exp so the backward pass can reconstruct the probabilities. BDH's problem is the reverse: its `(B, nh, L, N)` neuron tensors are the largest things in the model and are needed nowhere outside the block, so the neuron axis becomes a sequential grid dimension and never reaches HBM.

Two things worth knowing before using them:

- **They win memory traffic, not FLOPs.** Unstructured zeros still occupy a tensor-core lane. See the fairness note in `docs/ARCHITECTURE.md`.
- **A fully-masked query row is defined as zero**, not as flax's uniform-softmax-over-everything. Those rows are padded pack slots that `PointerHead` discards; flax's value for them depends on how far the key axis happens to be padded, which would make the same model give different numbers at different block sizes. `EMPTY_ROW_NOTE` in `kernels/cross_attention.py` has the detail.

`tests/test_kernels.py` asserts every kernel against a pure-JAX reference on values *and* on every gradient, and both fused arms against their reference arms under one shared parameter set. That is stricter than it may look necessary: the attention arm is the control every BDH result is measured against, so a wrong kernel there would not fail visibly, it would produce a plausible scaling curve that happens to be false, and a parameter-count test cannot catch it. On a CPU-only box the tests run under Pallas's `interpret=True`, which checks semantics but not lowering; set `KERNEL_INTERPRET=0` on a GPU or TPU to exercise the real thing.

## Parameter counts are derived

`count_params_analytic` builds N term by term from the config — layer norms, attention projections and their biases, MLPs, embeddings — and the test suite asserts it against the realised pytree at four widths. `train_model` re-checks it before every run and refuses to train on a mismatch.

This is `docs/PROJECT_PLAN.md` §3b's requirement, and it matters more here than in an ordinary model: an iso-parameter and an iso-FLOP comparison are genuinely different experiments, and neither is trustworthy if N is whatever `count_params()` happened to return after a library default changed.

Verified: analytic equals realised at d = 32, 64, 128, 256 and 512 (67,681 → 16,074,241 parameters), so width alone spans the grid's target range of roughly 0.5M to 50M.
