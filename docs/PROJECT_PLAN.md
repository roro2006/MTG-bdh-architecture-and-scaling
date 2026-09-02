# Project Plan

## 0. What this builds, and what is out of scope

**The deliverable is a working draft bot that can draft a set it was not trained on.** Everything else in this plan exists to produce that, or to produce it in a way that is understood rather than stumbled into.

Two commitments shape every stage below:

**Built from first principles.** The transformer front end, both interaction arms, the Pallas kernels for the attention-shaped and neuron-space operations, the parameter and FLOP accounting, and the scaling fit are all written here rather than imported. Where a library would do, the library is used only for operations whose memory behaviour is not ours to decide (LayerNorm, Dense, elementwise).

**Sized rather than guessed.** A Chinchilla-style law $L(N, D) = E + A/N^\alpha + B/D^\beta$ is fit on a cheap pilot grid, and the compute-optimal $(N^*, D^*)$ it yields is the configuration the shipped drafter is trained at. The scaling work is the sizing procedure, not a parallel research track.

**Out of scope for this version:** deck construction after the draft, win-rate prediction from `game_data_public`, in-draft signal reading beyond what the pool summarises, and multi-agent draft simulation. All are reasonable follow-ons; none is needed for a bot that picks well.

## 1. Task formalisation

At pick $t$: the current pack, the pool accumulated so far, and pack/pick number as scalar features are the inputs; the label is the card the human actually took (`pick` in 17lands' `draft_data_public`). Loss is cross-entropy computed only over the cards physically present in the pack — never the full vocabulary — which is what makes this a bounded decision task rather than an open-vocabulary one wearing a small vocabulary's clothes.

Splits are drawn on `draft_id`, not on individual rows. All 42 picks belonging to one draft land in the same split. A row-level split would leak: a withheld row's label sits inside the pool of the very next pick of the same draft, so the answer arrives through the pool even though the row itself was held out.

## 2. Data

Per-set `draft_data_public.<SET>.PremierDraft.csv.gz` files from 17lands. See `DATA.md` for what was verified directly about the format.

**Several sets, not one.** Cross-set generalisation is the goal, and it is not measurable with a single set ingested. Target: 6–10 sets, chosen to span mechanical variety rather than recency.

**Pack geometry must be inferred, not assumed.** Ingest currently hardcodes 14 selectable cards and 3 packs of 14 picks. Sets exist with other geometry, and the failure is quiet rather than loud: `PickData` validates the invariant $|\text{pool}| = \text{pack\_number} \times \text{picks\_per\_pack} + \text{pick\_number}$ on load, so a wrong constant makes every draft invalid and `on_invalid="drop"` discards the entire corpus without an error. Geometry is detected during ingest, persisted in `ingest_stats.json`, and read by `PickData` and `ModelConfig` — the pattern `card_feature_dim` already follows.

## 3. Representation

This is the stage the whole goal rests on, and it is where the previous version of the project was wrong.

### 3a. Every feature column must mean the same thing in every set

Colour identity, castable colours, mana value, type flags, rarity, power/toughness and `is_creature` already satisfy this. **Keyword flags do not.** They are fitted per set: on FIN, 84 of 118 Scryfall keywords occur on exactly one card and are dropped by `MIN_KEYWORD_CARDS`, leaving 34 columns whose meanings are particular to FIN. Column 41 is a different keyword in a different set, so a checkpoint is only interpretable alongside the exact table it trained against. That is disqualifying for a bot meant to draft an unseen set.

They are replaced by two things:

- **A fixed global keyword vocabulary** — a checked-in constant list, not fitted from the set at hand. Covers evergreen mechanics identically everywhere.
- **Structured mechanical features derived from oracle text** — roughly 80 columns pattern-matched over Scryfall's `oracle_text`: creates tokens, sacrifice outlet, cares about creatures dying, +1/+1 counters (makes and cares), graveyard recursion, self-mill, discard, draw, lifegain source and payoff, artifacts matter, enchantments matter, equipment and auras, tribal type reference, cares about attacking or blocking, ETB trigger, death trigger, activated ability with a mana cost, instant-speed interaction, removal by damage/destroy/exile/-X-X, counterspell, ramp and fixing, card selection.

### 3b. Structured features rather than a sentence embedding, and why

The obvious alternative is a frozen sentence encoder over the rules text. It is the wrong tool for this particular job, for a reason worth stating because it is not obvious: **sentence encoders are trained for semantic similarity, and synergy is complementarity.** "Sacrifice a creature: draw a card" and "Create two 1/1 tokens" are maximally synergistic and not remotely similar; "create two 1/1 tokens" and "create three 1/1 tokens" are similar and largely redundant. A similarity-shaped embedding space puts the wrong pairs near each other for this purpose.

Structured columns avoid that by construction: sacrifice-outlet and token-maker are separate features, so a bilinear form in the interaction arm learns their interaction in a single weight. They are also low-dimensional, set-independent by construction, and interpretable — which is what makes "why did it think these two cards go together" answerable.

A small projected text embedding can ride alongside later as a catch-all for mechanics the patterns miss. It is not the starting point.

### 3c. Three failure modes to design against

- **Name leakage.** Scryfall's `oracle_text` spells out the card's own name ("Whenever Zidane, Tantalus Thief attacks…"). Processing that raw reintroduces card identity as a feature — exactly the failure `MIN_KEYWORD_CARDS` exists to prevent, in continuous form. The name is stripped to a placeholder before anything reads the text.
- **Reminder text.** Parenthetical reminder text is redundant with the keyword flags and inflates apparent similarity between unrelated cards sharing a mechanic. Stripped.
- **Width inflating the small-$N$ corner.** `card_embedding` contains `_dense(F, embed_hidden)`, which is linear in $D$ while the rest of the model is quadratic. A wide table therefore inflates small models proportionally more than large ones, bending exactly the corner of the grid the Huber-on-log-residuals fit is most sensitive to. Total feature width stays under ~120 columns.

`CardFeatures.dense()` remains the single place the column layout lives.

## 4. Architecture

Full design in `ARCHITECTURE.md`. The three things that matter to this plan:

**Order is structurally unusable.** No positional encoding anywhere in the front end. Pack and pool are sets, and the architecture cannot use an order that was never there rather than being trained to ignore one.

**Synergy lives in the interaction arm, not the feature table.** Synergy is relational and cannot be encoded in a per-card vector. The arm — pack cards as queries, pool as context — is the bilinear form that connects a candidate's behaviour to the pool's. The feature table's job is only to make that connection *representable*.

**The output is closed to the pack.** The pointer head scores pack slots and softmaxes over just those.

Two interaction arms are implemented: cross-attention, and a from-scratch JAX port of BDH. **BDH is a candidate mechanism, not a research subject.** It ships if it drafts better. One consequence: `neuron_multiplier` no longer needs to be pinned at 4. That value existed solely to make an iso-parameter comparison possible; with the exponent comparison dropped, BDH is sized for how well it works, and larger neuron widths are back on the table — which is also the regime where its kernel earns its place (§5).

## 5. Kernel-level implementation scope

The commitment is that **attention-shaped and neuron-space operations are hand-written**; LayerNorm, Dense, softmax and elementwise ops stay in XLA, which already fuses them competently and whose gradients are not worth re-deriving to save nothing.

Current state against that commitment:

| | share of params | share of forward FLOPs |
|---|---|---|
| set encoders (pack + pool) — flax built-ins | 58% | — |
| interaction arm — hand-written Pallas | 39% | 26% (attention) / 37% (BDH) |
| everything outside the arm | — | 63–74% |

So the majority of the arithmetic still runs on `nn.MultiHeadDotProductAttention` (`set_encoder.py:49`). **Closing that gap — a Pallas kernel for the set encoder's masked, position-free self-attention — is a first-class task in the build order**, not an implied one. It is the same shape as the cross-attention kernel already written.

Two standing facts about the existing kernels, both of which belong in any writeup:

- **They win memory traffic, not FLOPs.** Unstructured zeros still occupy a tensor-core lane; a GPU multiplies by zero as fast as by anything else. Turning BDH's FLOP advantage into wall-clock needs block-structured sparsity, which is an architectural change rather than a kernel one.
- **They are correct but unmeasured.** Every kernel is asserted against a pure-JAX reference on values and on every gradient. None has run on real hardware — `default_interpret()` returns True off GPU/TPU, and interpret mode executes kernel semantics in pure JAX with no fusion at all. The first GPU session should re-run the kernel tests with `KERNEL_INTERPRET=0` and benchmark fused against reference across widths and neuron multipliers.

## 6. Sizing the drafter

A pilot grid is run purely to fit the law: 4–5 log-spaced widths, 3–4 log-spaced data fractions, one seed, both arms. Small enough to finish in an afternoon on one rented GPU.

The fit follows Chinchilla's robust procedure — Huber loss on log-residuals rather than least squares on raw loss, since raw least squares over-weights the small-$N$, high-loss corner — with bootstrapped confidence intervals on the exponents rather than bare point estimates.

Its output is a configuration, not a paper claim: given the compute budget available for the final run, $(N^*, D^*)$ says how wide the drafter should be and how much data it should see. Whichever arm's curve is better at that budget is the arm that ships.

**The $D$ axis must be data scale, not data repetition.** `--data-fraction` subsamples drafts, and grid cells use `--epochs` rather than `--steps`: at fixed steps a small fraction silently means many passes, and $\beta$ would then be measuring repetition. `run.py` warns past two passes.

## 7. Evaluation

**Per-pick breakdown is a first-class output, not a diagnostic.** The aggregate loss averages fourteen different problems. On FIN's val split, 7.1% of rows are one-card packs with loss identically zero and 21.4% have packs of three or fewer. Zero-loss picks are harmless to exponents — they scale $A$, $B$ and $E$ but leave $\alpha$ and $\beta$ alone — but picks 11–12 are a real hazard: easy without being trivial, so they saturate at small $N$ while hard picks keep improving, and a subset that stops responding to $N$ bends the aggregate curve in a way that reads as an exponent. `summarise_by_pick` reports all-picks and picks-0-8 side by side. **Headline numbers are the picks-0-8 slice.**

**Zero-shot protocol.** Train on $n-1$ sets, evaluate on the held-out set. Two reference points are required for the number to mean anything: a model trained directly on the held-out set (the ceiling) and the `pick_rate_prior` baseline (the floor). Expect a naive zero-shot figure to look decent for uninteresting reasons — rare-first and stay-on-colour carry a lot of draft accuracy, and the prior alone reaches 45.3% on FIN.

**Synergy probe.** Top-1 agreement cannot distinguish a model that learned card interaction from one that learned colour-matching. Two direct measurements, in `src/analysis/`:

- Hold the pack fixed, vary the pool, and report how much the candidate ranking moves. A model using synergy reorders; a colour-matcher barely does.
- Pool ablation: real pool versus a shuffled random pool of the same size. The accuracy gap is a scalar "how much does this model use its pool at all."

External validation against CubeCobra co-occurrence is the follow-up — cube curators pick cards for how they function together, so within-cube co-occurrence is a synergy signal not derivable from the model's own training data.

**Drafter metrics, not classifier metrics.** A tool that shows a ranked list is judged on top-3 as well as top-1, and on calibration. Both are reported.

## 8. Deliverables

- A trained drafter, at the size the fit chose, with a checkpoint and an inference entry point (`src/inference/`) that takes a pack and a pool and returns ranked picks.
- Zero-shot numbers on at least one fully held-out set, against both reference points.
- Synergy probe results, showing the model uses its pool.
- The fitted $L(N, D)$ curves for both arms, reported as the sizing procedure they are.
- Hand-written kernels covering all attention-shaped and neuron-space work, with GPU benchmarks against their references.
- A writeup carrying the derivations rather than just the results.

## 9. Risks worth naming up front

- **CPU-only training is the binding constraint on everything.** 561 examples/second means 2.3 hours per epoch at $d=64$ and makes the grid impossible. Nothing downstream is real until this is fixed, and the fix is cheap.
- **Synergy may be a thin signal in the data.** Human drafters at scale pick largely on card quality and colour; genuine synergy picks are a minority of decisions. The synergy probe exists to detect a null result honestly rather than to confirm a hoped-for one.
- **Structured mechanical features are hand-specified and therefore incomplete.** They will miss mechanics nobody thought to pattern-match. The probe and the held-out-set evaluation are what surface that.
- **Zero-shot numbers are easy to over-read.** Without the same-set ceiling and the prior baseline reported alongside, a plausible-looking figure means nothing.
- **BDH is not battle-tested.** A 2025 single-paper architecture with one reference implementation. Budget porting and debugging as research time, not translation.

## 10. Build order

1. **Get on a GPU.** Re-run both pilots unchanged to establish throughput. Re-run kernel tests with `KERNEL_INTERPRET=0`.
2. **Multi-set ingest**, with pack geometry inferred and persisted.
3. **Rebuild the feature table** — global keywords plus structured mechanical features, name-stripped, width-capped.
4. **Write the synergy probe** before the runs that would need it, so the representation change can be judged.
5. **In-distribution check:** one set, one width, old features versus new. If mechanical features do not beat keyword flags within a single set, they will not enable zero-shot either, and this costs minutes.
6. **Zero-shot probe** at one size: train on $n-1$ sets, test on the held-out one.
7. **Pilot grid and fit**, on the settled representation.
8. **Train the drafter** at $(N^*, D^*)$ with the winning arm.
9. **Set-encoder Pallas kernel**, closing the from-scratch gap in §5.
10. **Inference entry point and CLI**, then the writeup.

Steps 1–3 unblock everything. Step 4 before step 5 is deliberate: without the probe, step 5 can only tell you the loss moved, not why.
