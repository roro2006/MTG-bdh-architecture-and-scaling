# Models

Two architectures sharing one front-end (see `docs/ARCHITECTURE.md`):

- a set-attention encoder + pointer-network head, standing in for the "Transformer" arm of the scaling grid
- a JAX port of BDH's sparse/Hebbian block, dropped into the same front-end in place of the attention arm

Both consume identical composite card embeddings (color, cost, type, keyword-text) and identical permutation-invariant pack/pool encoding, so the only thing that differs between the two runs is the interaction mechanism itself.
