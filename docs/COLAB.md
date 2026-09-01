# Running the grid on free Colab

The full Cartesian grid described in `PROJECT_PLAN.md` §4 is 0.53 EFLOP.
On the CPU this project was developed on — measured at 25.6 GFLOP/s — that
is 240 days, so the grid does not exist as a CPU workload and no amount of
tuning changes that. This document is how it becomes an overnight job on a
free Colab T4 instead.

Two of the changes below cut wasted arithmetic and cost nothing. One cuts
the experiment itself and states what it gives up. One is hardware.

## Where it stands

| | cells | at 3 TFLOP/s (T4 fp32) | largest cell |
|---|---|---|---|
| full Cartesian | 120 | 49.1 h | 3.6 h |
| **L-shaped, forced rows dropped** | **62** | **26.5 h** | **3.76 h** |
| the same, in fp16 | 62 | 6.1 h | 0.87 h |

The pilot grid is 20 minutes in fp32 and under 5 in fp16.

Because the largest single cell fits inside a session at every setting,
**between-cell checkpointing is all the resumability that is needed** —
no cell has to survive being interrupted halfway.

## 1. Forced picks dropped from training — 7.14%, free

A pack holding one card admits one answer, so its cross-entropy is
identically zero for every parameter value and its gradient is zero with
it. On FIN that is 336,567 of 4,711,938 training rows — one in fourteen,
since each of the three packs ends in exactly one forced pick.

`decision_rows` removes them from the training stream and `run_cell` calls
it by default. This is not a trade: the rows carry no signal to lose, and
`test_a_forced_pick_contributes_no_gradient` asserts it on the gradient
rather than the loss, because a zero loss would also be produced by a model
that had merely learned those rows.

Evaluation keeps them. They are part of the task, `evaluate_by_pick`
reports them separately, and dropping them there would change what the
reported loss means.

The consequence worth carrying forward: **D now counts decisions, not
rows.** That is the honest definition anyway, because it matches both the
loss population the fit should consume and the population §6a concluded
the human-disagreement floor has to be measured over.

## 2. The L-shaped grid — 46%, one stated assumption

A full Cartesian product spends most of its compute where N and D are both
large, and that corner is the least informative per FLOP:

- **α** is fit by varying N at fixed D → the full size ladder, at one data fraction.
- **β** is fit by varying D at fixed N → the full fraction range, at a *cheap* size.

Neither exponent needs the intersection. `full_grid()` runs the size ladder
at `data_fraction=1.0`, the fraction sweep at the two cheapest widths, and
nothing in the expensive corner.

**What this assumes.** That the surface is separable — which is exactly
what `E + A/N^α + B/D^β` asserts by having no interaction term. The design
is therefore consistent with the model being fit, but it is *less able to
detect that the model is wrong.*

So `full_grid()` also emits three **interior points** at middling N and D,
one seed each. They are not used to fit either exponent. Their only job is
to be compared against the surface fitted without them: a systematic
residual there means the separable form is inadequate, and that is a
finding rather than a nuisance. They are cheap, and they are the difference
between assuming separability and having checked it.

This keeps both seeds and the full 0.5M–50M ladder — the two things the
plan's §8 says to sacrifice first. It turns out neither has to be.

## 3. Resumable cells — what makes a free runtime usable

Free Colab disconnects on a timer and the grid is longer than the timer, so
the grid has to survive its own runtime.

- Each cell writes **its own result file**, only after finishing. A file's
  presence therefore always means a completed cell, never an interrupted one.
- `run_grid` **skips** any cell whose file exists, so restarting is
  idempotent. Deleting a result file is how you ask for a cell to be rerun.
- Cells are ordered **most expensive first**, so a session that dies has
  done the costly work rather than saved it for last.
- Results go to **Drive**, not local disk, which Colab wipes on disconnect.

`test_completed_cells_are_skipped_and_results_reloaded` drives this end to
end and checks the result file's mtime, because a rerun that silently
retrained would otherwise return identical-looking results.

## 4. Hardware — the only change that matters

Everything above is a ~2x saving on a workload that is ~1000x too slow on
CPU. Get on a GPU first; do the rest once you are there.

`notebooks/colab_grid.ipynb` drives the whole thing. Two of its cells earn
their place:

**It asserts the GPU.** A JAX install that silently falls back to CPU costs
about 100x, and the grid then looks merely slow rather than misconfigured.

**It calibrates before it budgets.** Every hour-estimate here is a FLOP
roofline, and a roofline misprices anything bound by memory traffic or
kernel launch overhead rather than arithmetic — which the small cells are.
The notebook therefore runs one real cell, backs out *achieved* TFLOP/s,
and re-estimates the grid from the measurement. Treat the table at the top
of this document as a budget, not a prediction.

## Not done, and why

**Pool-length bucketing (37.8%).** Every pool is padded to 41 slots
regardless of its true size, and pool size is fully determined by
`pack_number` and `pick_number` — so 37.8% of every forward pass is
arithmetic on padding. Bucketing batches by pick number would recover most
of it.

It is not implemented because the FLOP-optimal choice and the wall-clock
optimal choice may genuinely disagree here: uniform shapes avoid
recompilation, ragged batches cost launches, and 41 is small enough that a
tile gets padded to a warp boundary anyway. Only a benchmark settles it,
and there is no GPU on the development machine. It would also move the
iso-FLOP axis of the fit, since `flops.py` deliberately counts padding
because a dense pass pays for it — so the accounting has to move with it
rather than after it.

**fp16.** The single biggest lever on a T4, whose fp16 tensor cores are
rated 8x its fp32 rate. Not wired in because it is a real numerics change —
fp16's narrow exponent range typically needs loss scaling — and shipping
untested numerics into a scaling study is how you get an exponent that is
wrong for reasons unrelated to the architecture. It deserves its own
before/after comparison on hardware.

Both are listed with measurements attached so the next person can pick them
up rather than rediscover them.
