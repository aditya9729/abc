# Original-methods ABC task results, 13 October 2026

Six native evaluations completed with clean parent, worker and host exits.
Each evaluation used the same captured scene, full initial INTEGRATION state,
initial RGB checks and 3,540 controls. Each used the native five-bottle judge.
No optimizer update ran during evaluation.

| Policy | Full task successes | VAL9100 bottles | VAL9101 bottles | VAL9102 bottles |
| --- | --- | --- | --- | --- |
| Original pi0.5 task SFT, selected step75 | 0/3 | 0/5 | 1/5 | 0/5 |
| Original EXPO learner, recovered evaluation state | 0/3 | 0/5 | 0/5 | 0/5 |
| Original FastTD3 scratch learner | Unavailable | Unavailable | Unavailable | Unavailable |

These are single trials on three captured native YAM scenes. They do not show
improvement, establish a reliable regression, or reproduce the papers.
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

## Recipe differences and current limits

The original EXPO learner and update equations remain unchanged.
Its task adapter uses YAM14 actions, three actor views and a five-bottle task.
The author's pick example uses DROID7 actions, two views and a cube task.
This experiment used three successful TRAIN demonstrations and three VAL episodes.
The author's collection example requests15 demonstrations; its online example uses10.

SFT ran at most100 updates. The selected checkpoint was step75.
It consumed4,800 windows across3,493 TRAIN anchors, about1.37 passes.
The run stopped at `update_limit`; it did not confirm a plateau.
Complete VAL sweeps must not be counted as TRAIN passes.
The author's pick SFT wrapper uses4,001 steps.
A fresh200-step preparation retains the original learner and validation stop rule.
It does not continue the previous optimizer state or reproduce the longer recipe.

EXPO collected11 complete episodes, then ran three update calls.
These calls produced60 critic/encoder and three actor/editor/temperature steps.
The author's online loop continues updates after eligible episodes.
The old ABC TRAIN run failed during host cleanup. Its status remains failed.
An authenticated recovery restored state for evaluation only.
The six new evaluations do not change the old failure status.

FastTD3's original actor, critics, distributional loss and update equations remain
unchanged. Batch32,768, replay51,200 per world and gamma0.99 remain fixed.
The declared64-world adaptation differs from the author's128 worlds.
The task adapter uses native current5/5 success plus bounded potential shaping.
The prior full training run was interrupted without a final export.
It supplies no accepted frozen performance score.

The full64 memory gate requires314,690,688 KiB under its conservative formula.
That estimate is not an observed simultaneous aggregate peak.
Current available RAM remains below the bound.
A separate public compiler-cache diagnostic measures two TRAIN ranks across13
resets per cache mode. It changes no learner or task math.
It compares actual physics bytes and task outputs in fresh CPU processes.
Its result cannot admit full64 training or supply a policy score.

## Viewer and owners

The local viewer at `http://127.0.0.1:8770/original-methods/` shows all six videos.
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
show the component boundaries. Native diagnostic results remain pending.
This explanation uses STE guidance. A full compliance check was not performed.
