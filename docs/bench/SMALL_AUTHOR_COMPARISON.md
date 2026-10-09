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

The physical task is `put_plastic_bottles_in_bin`. Released shared training-mixture metadata
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

ResFiT used 258 actual warmup controls and 32 training controls. It made
160 critic updates and 32 actor updates. Its 716 demo transitions produce
714 stored three-step replay rows. A single successful demonstration supplies
the action ranges and state statistics. It has three bottles; evaluation has five.
This data change affects the original action scaler. It can clip the base before
the residual acts. The tiny demo pool is not the original paper's expert dataset.
QF3 uses one complete warmup episode and one collection episode. It makes
128 critic updates and 16 actor updates. These interaction budgets differ.

Run each configuration with the installed coordinator:

```sh
ABC_BENCH_REPO=/home/user/aditya/RL/abc /home/user/aditya/RL/abc/.venv/bin/python -I -B -m abc_bench.runner --algorithm author-subset --checkpoint /home/user/aditya/RL/abc/cache/vla_abc130k_200000_v2.pt --training-config /home/user/aditya/RL/abc/configs/bench/author_subset_baseline.json --timeout-seconds 3600
```

Select `author_subset_resfit_train.json` for original-author training.
For another evaluation, bind its actual completed checkpoint and hash in
`author_subset_resfit_evaluate_supervisor.json`. The checked-in evaluation job
pins the checkpoint from this measured run. Select `author_subset_qf3.json`
for the QF3 diagnostic. Do not launch workers directly.

These jobs use existing normal installations. The native environment includes
original ABC model assets copied into its installed package. The complete
QF3 asset manifest contains 933 files, including the original `models/README.md`.
Keep this complete inventory after rebuilding the ABC wheel. Reinstalling the
wheel can remove separately copied assets. Its code and tokenizer remain in
the wheel. A failed inventory check must stop before model construction.

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
