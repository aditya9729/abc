# Original Real-Time EXPO-FT: fixed-data learning diagnostic

This test asks whether the original learner improves its value estimates on the
existing R1 Lite training data when collection time is removed. It is a named
schedule adaptation. It does not reproduce the author's online experiment.

The existing capacity worker gains an explicit `fixed_data_10` phase. The default
one-update phase remains available. The new phase calls the original learner
ten times, with measurements before learning, after one call, and after ten calls.
The original learner, replay sampler, losses, optimizers and action transforms
are unchanged. See [architecture](FIXED_DATA_EXPO_DIAGNOSTIC.mmd).

## Inputs and preserved settings

- RTC checkpoint 2000 initializes the actor. Seed 42 initializes the other models.
- The replay contains 7,069 demonstration rows and 9,899 recorded timeout rows.
  The 21 successful demonstrations share one source lineage.
- Batch size is 64. Each call performs 20 critic updates and one actor update.
  Candidate count is 32, action horizon is 16, execution length is 8, and delay is 5.
- The original replay restore runs before sampling. The existing queue and one
  pending batch remain intact during measurements.
- The worker has 2,300 seconds. The parent reserves 2,400 seconds, including cleanup.
  The job uses GPU 0. There is no automatic extension.

## Measurements

The calibration check uses 168 positive terminal windows from the demonstration
data. It reports each critic head, the mean prediction, head variance, and the
target critic. A falling average error can result from reduced disagreement
between heads while the mean prediction gets worse. Both quantities are reported.

For each demonstration, the ranking check freezes 32 original action candidates.
A separate copy inserts the logged successful eight-action chunk into candidate
31. The other 31 candidates have no outcome labels. The current encoder and target
critic rank these fixed banks using the same random keys and selected critic heads.
Selecting the reference does not establish that the other candidates would fail.

The measurements retain the original action prefix and normalization. They check
that model counters, sampler state and random-number state remain unchanged.
They do not test the updated actor's proposal distribution, autonomous recovery,
task success, or exact restart. No training checkpoint is saved by this diagnostic.

## Execution and evidence

The source was built and installed in separate canonical and numerical runtimes.
The installed checks passed 62 canonical and 39 numerical CPU test cases.
All 228 checked Python payloads matched the built source. The numerical runtime
retains the original dependency versions. These checks establish the caller's
behavior and installation, not learning performance.

The run `author-subset-20261015T232413-c92248` completed naturally in
1,596.806 seconds (26.6 minutes). The ten update calls took 1,171.538 seconds,
including compilation. Root verified the recorded results, unchanged inputs,
process exit and GPU release. An independent reviewer recomputed all measurements.

| Original update calls | Critic terminal MSE | Mean-prediction MSE | Variance between heads | Target critic MSE | Logged chunk ranked first | Mean logged-chunk rank |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.924788 | 0.200643 | 1.724144 | 1.924788 | 6/21 | 13.095 |
| 1 | 2.518605 | 2.504774 | 0.013832 | 1.948429 | 1/21 | 24.095 |
| 10 | 1.617648 | 1.615919 | 0.001729 | 1.754433 | 0/21 | 25.286 |

The final pooled critic error is below initialization because disagreement between
heads fell. The mean prediction is farther from the positive targets. All final
main and target predictions on these windows remain negative. The logged successful
chunk ranks worse. These observations do not establish useful learning, task harm,
convergence, or a failure cause. Other candidates have no outcome labels.

Across ten calls, the critic consumed 12,800 rows, including 117 positive-return
rows and 59 with zero critic weight. The actor consumed 640 rows from successful
trajectories, including 18 positive-return rows. These are recorded sample counts;
raw sampled labels were not saved for independent reconstruction. A zero immediate
return does not exclude a bootstrapped learning target.

The actor/editor/temperature counters reached 10; critic/filter/encoder counters
reached 200. Recorded state and returned metrics were finite. The sampler generated
16,640 critic rows and 704 actor rows; pending and queued batches explain the
unconsumed difference. All 399 saved arrays passed hash, shape and finite checks.

No policy rollout or trained checkpoint was produced. The candidate banks remain
fixed, so this test does not measure the updated actor's proposals. The next step
is verified correction/prevention data from independent failure origins, followed
by another bounded original-learner comparison. Longer training is not assumed to
resolve the observed gap.

Source review also found extra initial-agent and consumed-batch references in the
diagnostic caller. A separate lifetime correction is prepared. The completed run
is unchanged; no timing or numerical effect is attributed to those references.

[Root acceptance](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/fixed-data-learning-execution-20261015-02/ROOT_ACCEPTANCE.json) ·
[Independent final review](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/fixed-data-learning-execution-20261015-02/independent-actual-review/REVIEW.json).

- [Installed release](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/fixed-data-learning-release-01/RELEASE.json)
- [Source and installed review](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/fixed-data-learning-independent-review-20261015/RELEASE_REVIEW.json)
- [Bound job](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/fixed-data-learning-release-01/bound-01/JOB.json)
- [Execution directory](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/fixed-data-learning-execution-20261015-02)

Root owns installation, execution and acceptance. The implementation agent
prepared the caller. The independent reviewer checked source and recorded outputs.
This explanation uses STE guidance. Full ASD-STE100 compliance was not checked.
