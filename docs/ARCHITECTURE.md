# Architecture

## Why not a plain causal transformer

The obvious first move is to treat a draft like a sentence — feed the sequence of picks into a GPT-style causal block and predict the next one. That is the wrong shape for this problem, and not just as a style preference: within a single pack, and within the pool accumulated so far, order carries no information. A pack is a set of fourteen (or fewer) cards; whatever order the data pipeline happens to list them in is arbitrary. A causal transformer with positional encoding has no way to know that and will spend real capacity learning to ignore a signal that was never there. The architecture should be structurally incapable of using pack/pool order, not merely trained to ignore it.

There is a second issue with the sentence framing: a draft's entire history up to pick $t$ is already summarised by the pool at pick $t$. There is nothing left to recover by re-attending over raw pick history the way a language model re-attends over a token stream — the sufficient statistic is handed to the model directly at every step. Treating the draft as a long sequence to attend over is redundant work in exchange for nothing.

## The shared front end

Both interaction arms sit behind the same input pipeline, so that whatever difference shows up between them is attributable to the mechanism and nothing upstream of it.

### Cards are represented by behaviour, not identity

Each card's vector is built from its actual attributes rather than being one opaque row in a lookup table keyed by an arbitrary ID:

- **Set-independent attributes** — colour identity, castable colours, mana value, type flags, rarity, power/toughness, `is_creature`.
- **A fixed global keyword vocabulary** — a checked-in constant list covering evergreen mechanics, identical for every set.
- **Structured mechanical features derived from oracle text** — roughly 80 columns pattern-matched over Scryfall's rules text: creates tokens, sacrifice outlet, cares about creatures dying, +1/+1 counters, graveyard recursion, discard, artifacts matter, removal, ramp, and so on.

Three consequences follow, and all three are load-bearing rather than incidental:

- **A set released after training gets a usable representation immediately**, from attributes alone. An ID-embedding table cannot do this at all, and a per-set-fitted keyword block cannot either.
- **What a card mechanically does becomes representable.** "Sacrifices a creature" and "creates tokens" are distinct columns rather than both being invisible, which is the precondition for the model learning that they belong together.
- **The residual is interpretable.** Colour, cost, type and mechanical behaviour are handed to the model for free, so whatever internal structure it still builds is by construction the part those features do not explain.

**Nothing is fitted per set.** An earlier version of this project derived keyword columns from the set at hand, which meant column 41 was a different keyword in a different set and a checkpoint was only meaningful with the exact table it trained against. See `PROJECT_PLAN.md` §3 for why that had to go and what replaced it.

### Permutation-invariant set encoding

Pack and pool are each run through a Set Transformer-style encoder — attention blocks with no positional encoding, pooled into a single representation per set. No position, no order, nothing for the architecture to overfit to.

### Pointer-network output

The model scores only the cards physically present in the current pack and takes a softmax over just those scores. It is structurally unable to "pick" a card outside the pack — the same guarantee against naming a nonexistent choice that constraining a vocabulary buys on the input side, arrived at through the output layer instead.

## Where synergy lives

This is the design decision the whole bot rests on, so it is worth stating plainly: **synergy cannot live in the feature table.**

Synergy is a relation between two cards, and a per-card feature vector has nowhere to put a relation. What the feature table can do — and all it has to do — is make the relation *representable*: carry enough about what each card mechanically does that a learned bilinear map can recover the interaction.

The interaction arm is where the relation is actually computed. Pack cards are queries, the pool is context, and the arm produces a per-candidate compatibility score. `W_q` and `W_k` learning to map "sacrifice outlet" in the query onto "token maker" in the key *is* a synergy measurement. Pack and pick number are folded in on the query side as a small learned feature — "how far into the draft am I" genuinely changes what a good pick looks like — not as a causal position.

This is also why structured mechanical columns beat a frozen sentence embedding here. Sentence encoders are trained for semantic **similarity**; synergy is **complementarity**. "Sacrifice a creature: draw a card" and "Create two 1/1 tokens" are maximally synergistic and not similar at all, while "create two 1/1 tokens" and "create three 1/1 tokens" are similar and largely redundant. With separate columns, one weight in the bilinear form expresses the interaction; with a similarity-shaped embedding, the arm has to first undo the geometry it was handed.

**Top-1 accuracy cannot tell you whether any of this worked.** A model that only learned colour-matching scores well. The probes in `src/analysis/` — hold the pack fixed and vary the pool; compare a real pool against a shuffled one — are what distinguish the two.

## Where the two arms diverge

Composite embeddings, set encoding and the pointer head are identical between runs. The one thing that changes is the interaction mechanism between pool and pack:

- **Attention arm** — the cross-attention block described above.
- **BDH arm** — BDH's sparse, Hebbian-plasticity block in the same position, consuming the same pool/pack representations and producing the same shape for the pointer head.

No JAX implementation of BDH existed; the reference (`pathwaycom/bdh`) is a bare PyTorch script. The port lives in `src/models/bdh_arm.py`.

**Two things in the reference had to go**, for the reason this document already gives. The reference is a causal language model: it applies RoPE to its query/key features and masks its scores with `tril(diagonal=-1)`. Both encode token order. A pool is a set, so there is no order for them to encode, and keeping them would have contradicted "Why not a plain causal transformer" above. The port keeps everything that makes BDH BDH — the wide ReLU neuron space, the absence of a softmax, the Hebbian accumulation, the multiplicative gate, the low-rank encode/decode — and drops the two pieces that are about sequences. `tests/test_kernels.py` asserts the result is exactly invariant to permuting the pool, which is the property that would break first if order crept back in.

**BDH is a candidate mechanism, not a research subject.** It ships if it drafts better. That has a concrete consequence for sizing: `neuron_multiplier` was pinned at 4 solely to make an iso-parameter comparison possible against a cross-attention block (12·D² against 12·D² + 15·D — matching to 0.5% at D=256). With the exponent comparison dropped, that constraint is gone. BDH can be sized for how well it works, and the reference's much larger neuron widths are back on the table — which is also the regime where its kernel does what it was designed to do (see below).

## Compute accounting

Parameter counts and FLOP counts are derived term by term (`count_params_analytic`, `src/models/flops.py`) and asserted against the realised model, rather than read off whatever the framework reports. Both numbers feed the sizing decision in `PROJECT_PLAN.md` §6, and a sizing decision made on a number nobody verified is a guess wearing a derivation's clothes.

### What the FLOP accounting showed

Deriving BDH's FLOP count term by term (`flops.py::_bdh_layer`) produced a result that should be read before anyone writes "BDH does less arithmetic" anywhere.

**Sparsity can only skip two of six terms.** A BDH layer spends its arithmetic on three `D → N` encodes, an interaction score, a value matmul, and an `nh·N → D` decode. Only the score and the decode reduce over the neuron axis against sparse operands, so only they shrink with density. The three encodes are paid in full at *any* density, for an unavoidable reason: you cannot know a ReLU will output zero without first computing its input. That is not an implementation limitation, it is the shape of the computation.

At `D=256` and iso-parameter sizing:

| | forward FLOPs, arm only |
|---|---|
| attention arm | 59.9M |
| BDH arm, dense | 91.7M (1.53× attention) |
| BDH arm, ideal at 25% density | 79.0M (86% of dense) |
| BDH arm, ideal at 2% density | 75.0M (82% of dense) |

So at that sizing BDH starts 53% *more* expensive than the arm beside it, and perfect sparsity exploitation would claw back at most about 18%. The floor is the encodes. **A sparsity-based efficiency claim does not survive contact with this architecture at iso-parameter sizing.** `bdh_ideal_flops` computes the bound for any measured density, and `measure_density` supplies density from a real batch rather than an assumption — at initialisation it is ~0.5, a property of the initialiser that says nothing about a trained model.

## Kernel scope

**Attention-shaped and neuron-space operations are hand-written in Pallas. LayerNorm, Dense, softmax and elementwise ops stay in XLA**, which fuses them competently and whose gradients are not worth re-deriving to save nothing. That is the same line FlashAttention itself draws: hand-write where the memory-hierarchy decision is yours.

The two arms already meet it, and they solve opposite problems. Cross-attention is small enough that a whole `(batch, head)` slice fits in SRAM — pack ≤ 14, pool ≤ 42 — so the kernel skips FlashAttention's tiling and online softmax entirely, doing forward and backward in one block each and keeping only the log-sum-exp so the backward can reconstruct the probabilities. BDH's problem is the reverse: its `(B, nh, L, N)` neuron tensors are the largest things in the model and are needed nowhere outside the block, so the neuron axis becomes a *sequential grid dimension*, tiled and accumulated in SRAM, never reaching HBM.

The backward pass is deliberately two kernels rather than one. Activation gradients and weight gradients reduce along opposite axes — `dxq` sums over heads and neuron tiles, `d_enc` sums over the batch. Doing both in one kernel would need float atomics, which accumulate in nondeterministic order, and run-to-run reproducibility is not negotiable when runs are being compared against each other. Two kernels with different grids, each reducing over a sequential axis it owns: slightly more recomputation, exactly reproducible.

### The gap

The commitment is not yet met. `nn.MultiHeadDotProductAttention` is still used at `set_encoder.py:49`, and the set encoders are the larger part of the model:

| | share of params | share of forward FLOPs |
|---|---|---|
| set encoders — flax built-ins | 58% | — |
| interaction arm — hand-written | 39% | 26% (attention) / 37% (BDH) |
| everything outside the arm | — | 63–74% |

A Pallas kernel for the set encoder's masked, position-free self-attention closes it. It is the same shape as the cross-attention kernel already written, and it is an explicit item in the build order rather than an aspiration.

### Two standing caveats

- **Kernels win memory traffic, not arithmetic.** Unstructured zeros still occupy a lane in a tensor-core tile, so a GPU multiplies by zero exactly as fast as by anything else. Anything reported on a wall-clock axis should say which of the two it means. Realising BDH's FLOP advantage as time needs block-structured sparsity — an architectural change, not a kernel one.
- **They are correct but unmeasured.** Every kernel is asserted against a pure-JAX reference on values *and* on every gradient, and both fused arms against their reference arms under one shared parameter set. But `default_interpret()` returns True off GPU/TPU, and interpret mode runs kernel semantics in pure JAX with no fusion — so nothing here has a performance number attached yet. Set `KERNEL_INTERPRET=0` on real hardware to exercise the lowering.
