# Architecture

## Why not a plain causal transformer

The obvious first move is to treat a draft like a sentence — feed the sequence of picks into a GPT-style causal block and predict the next one. That's the wrong shape for this problem, and not just as a style preference: within a single pack, and within the pool accumulated so far, order carries no information. A pack is a set of fifteen (or fewer) cards; whatever order the data pipeline happens to list them in is arbitrary. A causal transformer with positional encoding has no way to know that and will spend real capacity learning to ignore a signal that was never there. The architecture should be structurally incapable of using pack/pool order, not merely trained to ignore it.

There's a second issue with the sentence framing: a draft's entire history up to pick $t$ is already summarized by the pool at pick $t$. There's nothing left to recover by re-attending over raw pick history the way a language model re-attends over a token stream — the sufficient statistic is handed to the model directly at every step. Treating the draft as a long sequence to attend over is redundant work in exchange for nothing.

## The shared front-end

Both architectures in this project — the attention arm and the BDH arm — sit behind the same input pipeline, so that whatever difference shows up in the scaling curves is attributable to the interaction mechanism and nothing upstream of it.

**Composite card embeddings.** Each card's vector is built from its actual attributes rather than being one opaque row in a lookup table keyed by an arbitrary ID: color identity (5-dim multi-hot), mana value, card type, and a small embedding derived from oracle-text keywords, summed or concatenated into one vector. Two consequences follow from this, and both matter for the project's goals rather than being incidental:

- A card released after training gets a reasonable embedding immediately, from its attributes, instead of needing a new vocabulary slot and retraining. An ID-embedding table can't do this at all.
- Color, cost, type, and keyword text are handed to the model for free. Whatever internal structure the model still has to build in order to improve its predictions is, by construction, the part that isn't explained by those trivial features — which is exactly the residual a later interpretability pass over "why does the model think these two cards go together" needs to be looking at.

**Permutation-invariant set encoding.** Pack and pool are each run through a Set Transformer-style encoder — attention blocks with no positional encoding, pooled into a single representation per set (Lee et al., 2019's induced-set-attention approach is the natural fit here, though plain sum/mean pooling after a shared per-card MLP is a reasonable simpler baseline to start from). No position, no order, nothing for the architecture to overfit to.

**Cross-attention between pool and pack.** The decision at each pick is "given what I already have, which of these options fits" — pool as context, each pack card as a query (or the reverse), producing a per-candidate compatibility score. Pack/pick number is folded in here as a small learned feature on the query side, not as a causal position.

**Pointer-network output.** The model scores only the cards physically present in the current pack and takes a softmax over just those scores. It is structurally unable to "pick" a card outside the pack — the same guarantee against hallucinating a nonexistent choice that the original addition-transformer exercise got from constraining its vocabulary to digits, just arrived at through the output layer this time instead of the input.

## Where the two architectures diverge

Composite embeddings, set encoding, and the pointer output head are identical between runs. The one thing that changes is the interaction mechanism sitting between the pool and pack representations:

- **Attention arm** — the cross-attention block described above.
- **BDH arm** — BDH's sparse, Hebbian-plasticity block substituted in the same position, consuming the same pool/pack representations and producing the same shape of output for the pointer head to score against.

No existing JAX implementation of BDH exists; the reference implementation (`pathwaycom/bdh`) is a bare PyTorch script. The port lives in `src/models/bdh_arm.py`.

**Two things in the reference had to go, and both for the reason this document already gives.** The reference is a causal language model: it applies RoPE to its query/key features and masks its scores with `tril(diagonal=-1)`. Both encode token order. A pool is a set, so there is no order for them to encode, and keeping them would have contradicted "Why not a plain causal transformer" above — the same commitment that rules out a causal transformer for the attention arm rules out a causal BDH. The port keeps everything that makes BDH BDH (the wide ReLU neuron space, the absence of a softmax, the Hebbian accumulation, the multiplicative gate, the low-rank encode/decode) and drops the two pieces that are about sequences. `tests/test_kernels.py` asserts the result is exactly invariant to permuting the pool, which is the property that would break first if order ever crept back in.

**Sizing had to change too.** The reference's `mlp_internal_dim_multiplier=128` gives a neuron width of `N = 32*D`, or roughly 25M parameters for one layer at `D=256` — against 0.8M for a cross-attention block. No iso-parameter grid is possible at that ratio. `neuron_multiplier=4` makes a BDH layer cost `3·m·D² = 12·D²` against a cross-attention block's `12·D² + 15·D`: iso-parameter to within a term linear in D, and 1,572,864 against 1,581,312 at `D=256` in practice — a 0.5% gap on the arm and 0.2% on the model total.

## A fairness note for the scaling comparison

Matching the two arms on parameter count ($N$) is not the same as matching them on compute, precisely because BDH's activations are sparse by design — a forward pass can do meaningfully less arithmetic than a dense attention block with the same number of weights. Reporting only an iso-parameter curve would let "which architecture scales better" quietly mean two different things depending on which axis is held fixed. Both an iso-parameter and an iso-FLOP comparison are reported for exactly this reason — see `PROJECT_PLAN.md` §3d and §5.

### What the FLOP accounting actually showed

Deriving BDH's FLOP count term by term (`src/models/flops.py::_bdh_layer`) changed the picture, and the numbers below should be read before anyone writes "BDH does less arithmetic" in a results section.

**Sparsity can only skip two of the six terms.** A BDH layer spends its arithmetic on three `D → N` encodes, an interaction score, a value matmul, and an `nh·N → D` decode. Only the score and the decode reduce over the neuron axis against sparse operands, so only they shrink with density. The three encodes are paid in full at *any* density, for an unavoidable reason: you cannot know a ReLU will output zero without first computing its input. That is not an implementation limitation, it is the shape of the computation.

At `D=256` and the iso-parameter sizing, the consequences are stark:

| | forward FLOPs, arm only |
|---|---|
| attention arm | 59.9M |
| BDH arm, dense | 91.7M (1.53× attention) |
| BDH arm, ideal at 25% density | 79.0M (86% of dense) |
| BDH arm, ideal at 2% density | 75.0M (82% of dense) |

So at this sizing BDH starts 53% *more* expensive than the arm it is being compared against, and perfect sparsity exploitation would claw back at most about 18%. The floor is the encodes. **A sparsity-based efficiency claim does not survive contact with this architecture at iso-parameter sizing**, and the honest iso-FLOP comparison is between the dense counts.

That does not make the comparison uninteresting — it means the interesting question is quality per parameter and per dense FLOP, not a sparsity dividend. If a sparsity dividend is wanted, it needs either a much larger `neuron_multiplier` (where the encodes stop dominating, but iso-parameter is lost) or block-structured sparsity, which is an architectural change rather than a kernel one. `bdh_ideal_flops` computes the bound for any measured density, and `measure_density` supplies the density from a real batch rather than an assumption — at initialisation it is ~0.5, which is a property of the initialiser and says nothing about a trained model.

### Dense FLOPs are what the hardware runs

Both arms have fused Pallas kernels (`src/models/kernels/`). They win **memory traffic**, not arithmetic: unstructured zeros still occupy a lane in a tensor-core tile, so a GPU multiplies by zero exactly as fast as by anything else. Anything reported on a wall-clock axis should say which of the two it means.

The kernels exist for both arms deliberately. The attention arm is the control; hand-optimising only BDH would mean any wall-clock difference measured kernel effort rather than architecture.
