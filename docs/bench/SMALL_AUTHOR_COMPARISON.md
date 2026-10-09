# Small ABC-VLA comparison

This experiment compares the same local ABC-VLA checkpoint in three conditions:
frozen, an unchanged author ResFiT learner with an ABC adapter, and QF3 implemented
from its paper. The official QF3 project page does not yet link author code. Its
exact ABC initializer and several critic settings remain unresolved.

Use three development layouts: 9011, 9012 and 9013. Use the original one-world
ABC MJWarp simulator, five bottles, fixed mesh variants and scales, raw RGB
224×168 cameras, five flow steps, and fifteen controls per thirty-control plan.
Use the same placement-history success rule, plus the original current-success
metric. Each evaluation episode has at most 1,000 controls. Train on other layouts.
These settings select one explicit comparison; they do not establish paper results.

The physical task is `put_plastic_bottles_in_bin`. Released checkpoint metadata
maps that scene to `sim throw plastic bottles in bin`. This is the recorded
training prompt remap. The physical throw scene is a different task.

The original ResFiT source stays at `66262b06`. The ABC simulator and VLA source
stay at `d0832d12`. Boundary code converts camera layouts and passes physical
joint angles and gripper controls. It does not replace the author learner.
The original action scaler can change the base action even with zero residual.
Evaluation reports that difference instead of assuming zero-residual equivalence.

Two raw TRAIN episodes are already local. The original task judge accepts the
three-bottle trajectory at frame716. The five-bottle trajectory never succeeds.
Only the real successful prefix is used for the ResFiT demonstration pool.
CPU FFmpeg decoding is explicit. Decoder parity with TorchCodec is not claimed.

Training counts are reduced for this diagnostic. ResFiT keeps the original
128-online/128-demo batch, three-step replay and four-critic/one-actor cadence.
QF3 keeps its filtered Q-gradient, flow-matching and frozen-base anchors, rank4
head LoRA, and causal chunk Bellman target. Its unresolved critic settings are
explicit local choices. The two methods have different training costs. A small
sample can show execution and observed outcomes; it cannot establish a reliable
improvement or a full paper reproduction.

Root runs GPU0 jobs through `python -m abc_bench.runner --algorithm author-subset
--training-config JOB.json`. Each job has an explicit interpreter, module,
arguments and deadline. The shared lease, occupancy check and original ledger
remain active. The root process reaps its whole worker group before releasing
the lease. Other GPU jobs remain outside this process group.

Reusable launch code is [subset.py](../../abc_bench/subset.py). The ResFiT runtime
supervisor is [subset_worker.py](../../abc_bench/subset_worker.py). Method code
and data adaptation remain in their separate installed packages. Run artifacts
are under `abc/outputs/bench/author-subset-*`. They contain the resolved job,
execution logs and actual method receipt. Missing results stay unavailable.

The coordinator assigns implementation tasks and owns execution. The host owner
connects original ABC inference and physics. The ResFiT owner verifies real
demonstrations and author updates. The QF3 owner checks paper equations and
the runnable learner. Root and peers review the other owners' changes.

[Architecture source](SMALL_AUTHOR_COMPARISON.mmd) shows the package boundaries.
This explanation uses STE guidance. It has not had a full compliance check.
