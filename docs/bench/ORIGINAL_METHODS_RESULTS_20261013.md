# ABC adaptation results, 13 October 2026

**Status correction, 18:05 UTC:** the standard EXPO campaign was stopped for
method reassessment. The user requires Real-Time EXPO-FT. No Real-Time EXPO-FT
training or evaluation has run here. The results below are historical ABC
adaptations, not reproductions of either paper. See the current
[recipe audit](RECIPE_FIDELITY_AUDIT.md).

Run `author-subset-20261013T171702-e41a5f` stopped during warmup with zero
learner updates and no accepted export. Three episodes completed. The final
host counted 11,433 controls; the stale worker receipt counted 11,428.
The cancellation does not measure RL learning. Its parent, worker and host
are gone, and the coordinator released its GPU lease.

Nine native evaluations completed with clean parent, worker and host exits.
Each evaluation used the same captured scene, full initial INTEGRATION state,
initial RGB checks and 3,540 controls. Each used the native five-bottle judge.
No optimizer update ran during evaluation.

| Policy | Full task successes | VAL9100 bottles | VAL9101 bottles | VAL9102 bottles |
| --- | --- | --- | --- | --- |
| Original pi0.5 task SFT, selected step75 | 0/3 | 0/5 | 1/5 | 0/5 |
| Original EXPO learner, recovered evaluation state | 0/3 | 0/5 | 0/5 | 0/5 |
| Fresh pi0.5 task SFT, selected step125 | 0/3 | 4/5 | 1/5 | 1/5 |
| Original FastTD3 scratch learner | Unavailable | Unavailable | Unavailable | Unavailable |

These are single trials on three captured native YAM scenes. The new SFT actor
made more partial progress on two scenes. No policy completed the full task.
The new result follows supervised training. It does not establish an EXPO RL
gain, a statistically reliable improvement, or a paper reproduction.
They are not R1Lite results. Scratch state-based FastTD3 and pretrained RGB
EXPO compare training strategies; this comparison does not isolate algorithm quality.

The pre-RL canonical run identifiers are `060006-ed0bda`, `061838-1042ed`
and `063351-86e70f`. The post-RL identifiers are `045029-840400`,
`051634-3410b3` and `054017-f863f3`. Each has prefix
`author-subset-20261013T` in `outputs/bench`.
The authenticated V3 matched summary is
`builds/abc-expo-fast-matched-evaluation-20261013/root-verification-v3-precase3/ACTUAL_SUMMARY.json`
relative to this repository's parent. Its SHA256 is
`d455f98589a92eb2f79a08ab53d5665519e42b3377a1ae997763d5da61955009`.
The root acceptance and browser proof are in that same directory.

The selected125 canonical runs are `153102-cd24cf`, `154656-5e19cf` and
`160338-d053e9`, with the same full prefix above. Their root terminal records are
under `builds/expoft-sft125-captured-evaluation-20261013/actual-9100-launch`,
`actual-9101-launch` and `actual-9102-launch`, relative to this repository's parent.
The separate V4 reader uses an explicit selected125 manifest. It preserves the
original producer, frozen-weight, captured-state, RGB and complete-episode checks.
V3 and the two earlier cohorts remain unchanged. No matching post-RL125 cohort exists.

## Recipe differences and current limits

The original EXPO learner and update equations remain unchanged.
Its task adapter uses YAM14 actions, three actor views and a five-bottle task.
The author's pick example uses DROID7 actions, two views and a cube task.
This experiment used three successful TRAIN demonstrations and three VAL episodes.
The author's collection example requests15 demonstrations; its online example uses10.

The earlier SFT run used at most 100 updates and selected step75.
That checkpoint consumed 4,800 windows across 3,493 TRAIN anchors, about 1.37 passes.
The run stopped at `update_limit`; it did not confirm a plateau.
Complete VAL sweeps must not be counted as TRAIN passes.
The author's pick SFT wrapper uses4,001 steps.
A fresh run from the original base used a maximum of 200 updates. It stopped at
175 for `heldout_macro_loss_plateau` and selected update125. Each complete VAL
sweep covered 3,324 windows. Selection required 1% relative improvement, with
patience two and a minimum of 50 updates. Selected loss was 0.02018792, versus
0.04678183 initially. This loss measures demonstration agreement, not task success.
The learner, three TRAIN demonstrations, statistics, batch64, seeds and learning
rate stayed fixed. This was a fresh restart, not optimizer continuation or the
author's longer recipe. Canonical producer: `134344-21980e`.

EXPO collected11 complete episodes, then ran three update calls.
These calls produced60 critic/encoder and three actor/editor/temperature steps.
The author's online loop continues updates after eligible episodes.
The old ABC TRAIN run failed during host cleanup. Its status remains failed.
An authenticated recovery restored state for evaluation only.
The clean frozen evaluations do not change the old failure status.

FastTD3's original actor, critics, distributional loss and update equations remain
unchanged. Batch32,768, replay51,200 per world and gamma0.99 remain fixed.
The declared64-world adaptation differs from the author's128 worlds.
The task adapter uses native current5/5 success plus bounded potential shaping.
The prior full training run was interrupted without a final export.
It supplies no accepted frozen performance score.

The completed scene-cache resource run `152212-e1c244` tested six fresh processes.
Each process performed 13 resets and 26 controls. Retained scene limits were
12, 0 and 1, on TRAIN ranks0 and55. All four recorded model, state, action, reward
and judge comparisons matched. Raw XML differences are reported separately.

| TRAIN rank | Peak RAM, limit12 | Peak RAM, limit0 | Peak RAM, limit1 |
| --- | --- | --- | --- |
| 0 | 6,738,728 KiB | 4,313,200 KiB | 4,778,356 KiB |
| 55 | 6,577,800 KiB | 4,010,416 KiB | 4,570,176 KiB |

Limit0 reduced these separate process peaks by 36–39%. The unchanged formula,
64 times the maximum peak times 1.25 plus 8 GiB, still requires 353,444,608 KiB.
The root check observed 227,061,908 KiB available. This does not admit training.
It is an estimate, not a simultaneous aggregate measurement or a policy score.

The corrected default-off unused-mesh candidate completed native run
`164627-a64c76` in 110.124 seconds. Four fresh processes ran 52 controls.
Rank0 covered 12 resets; rank55 covered its heavy initial reset. Both comparisons
passed the explicit XML deletion check, retained mesh geometry, remaining physical
arrays/options, full INTEGRATION and native state/action/reward/judge checks.
Raw XML and complete array inventories differ by design. This is not byte-identical
MJB evidence. The earlier `162141-ead7ca` failure remains failed: it rejected
unnamed mesh declarations in the original input. Version0.1.3 retains those
declarations unchanged and removes only explicitly named, unreferenced meshes.

| TRAIN rank | Unpruned peak RAM | Pruned peak RAM | Reduction |
| --- | --- | --- | --- |
| 0 | 3,632,804 KiB | 2,896,732 KiB | 20.3% |
| 55 | 3,151,260 KiB | 2,518,576 KiB | 20.1% |

Both variants used scene limit0 and the unchanged compiler cache. These are
within-test comparisons, not an additive savings claim across earlier tests.
The unchanged 64-world formula now estimates 240,127,168 KiB, below the
279,913,112 KiB available at root review. No simultaneous 64-world test or
training ran. Production wiring, a fresh source-matched resource proof and a
64-world qualification remain required. Root evidence is
`builds/abc-fasttd3-implicit-mesh-retention-20261013/actual-native-launch/ACTUAL_TERMINAL_RESULT.json`
relative to this repository's parent. Worker receipt SHA256:
`fa12cc04b6dbfc96fa2edc8c074e7037b2a0f82111e5c1b9e20d6c14449e48c5`.
The diagnostic covers rank0 resets0..11. It does not cover the final autoreset
after twelve full episodes. The earlier scene-cache test reached its largest
rank0 peak at reset12. The production proof must include that transition before
using its peak for memory admission. The estimate above is incomplete lifecycle
coverage; it does not establish that the full campaign fits.
The original learner, reward and full64 guard are unchanged.
The original ABC0.1.0 captured host remains installed. Resource candidates use
separate runtime prefixes; they cannot bypass the original capture contract.

FastTD3 already learns its own twin value functions. No external value model is
missing. This adapter uses the author's HumanoidBench defaults, not a published
ABC task recipe. At 0.034 seconds/control, gamma0.99 gives a 3.38-second e-fold
horizon. Successful TRAIN demonstrations take 35–48 seconds. Sparse five-bottle
success, reward scale and exploration are therefore task-fit concerns.
Gamma alone is not a proven bug: original author tasks also use short horizons,
with different rewards. Potential shaping preserves the discounted objective;
it does not extend the horizon or provide successful replay. Finite update
gradients do not establish useful learning. More training is not a proven fix.
The source-bound audit is
`builds/abc-fasttd3-task-design-read-only-20261013/NOTE.md`, relative to this
repository's parent. It separates reward arithmetic from learning performance.

## Viewer and owners

The local viewer at `http://127.0.0.1:8770/original-methods/` shows the separate
policy cohorts, accepted episode videos and measured resource cases.
Use the normal installed export CLI described in
[ORIGINAL_METHODS_VIEWER.md](ORIGINAL_METHODS_VIEWER.md).
Its export preserves existing results and videos.
An unchanged data refresh preserves the playing video element.

Root owns native runs, runtime releases and result publication.
The preparation agents own isolated candidates and CPU evidence.
The review agent checks the frozen source independently.
External review was waived by the user's results-first instruction.
Leela's jobs and other GPUs remain untouched.

[Viewer architecture](ORIGINAL_METHODS_VIEWER.mmd) and
[resource diagnostic architecture](FASTTD3_COMPILER_CACHE_DIAGNOSTIC.mmd)
show the component boundaries. The scene-cache and unused-mesh diagnostic
measurements are complete. FastTD3 production integration and training results
remain pending. No new EXPO RL result exists for the selected125 policy.
This explanation uses STE guidance. A full compliance check was not performed.
