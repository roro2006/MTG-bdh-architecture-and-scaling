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

No existing JAX implementation of BDH exists; the reference implementation (`pathwaycom/bdh`) is a bare PyTorch script. Porting it is real work and gets its own acceptance test — training it on a small toy task and confirming it actually converges and that its sparsity/positivity properties hold in this implementation — before it's allowed anywhere near the full scaling grid. See `PROJECT_PLAN.md` §3 for how that's staged.

## A fairness note for the scaling comparison

Matching the two arms on parameter count ($N$) is not the same as matching them on compute, precisely because BDH's activations are sparse by design — a forward pass can do meaningfully less arithmetic than a dense attention block with the same number of weights. Reporting only an iso-parameter curve would let "which architecture scales better" quietly mean two different things depending on which axis is held fixed. Both an iso-parameter and an iso-FLOP comparison are reported for exactly this reason — see `PROJECT_PLAN.md` §3d and §5.
