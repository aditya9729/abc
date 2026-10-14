# Original RL recipe experiments — 14 October 2026

We have run the original FastTD3 and Mulligan programs on their supported tasks. We have also run the original Real-Time EXPO learner through a declared Catcher integration. The remaining active run evaluates the locally trained Mulligan checkpoint on all 8,000 locked starts. Its final score is pending.

[Open the results and complete recordings](http://127.0.0.1:8770/original-methods/). These are separate task experiments. They do not rank the three algorithms against each other.

## Measured results

| Experiment | Executed work | Measured outcome | Main limit |
| --- | --- | --- | --- |
| Original FastTD3, H1Hand Hurdle | Current author recipe, seed 1, 128 worlds, 100,000 vector calls; 12.8 million training transitions | Last periodic return: **572.72 at 95,000 calls**. Released curve at the same nominal coordinate: **778.71** | One seed; historical recipe metadata is incomplete. Nine printed physics warnings. No final 100,000-call evaluation |
| Original Mulligan, Square-Narrow R1 | Original IDQL and frozen-actor DIVL stages, **150,000 updates each**, using the released data | On the same 50 starts with 32 candidates: **IDQL 49/50; DIVL 47/50**. Full 8,000-start DIVL result: **pending** | Small cohort is reused development data. Full evaluation uses one training seed and explicit evaluation/reset seeds |
| Real-Time EXPO, Catcher integration | Released learner and asynchronous action sampler; **3,343 actor/editor updates and 66,860 critic/filter updates** | Frozen policy: **3/4**, starting policy: **1/4**, on the same four environment seeds | Declared task/backend adaptation; one layout and four reused starts. Training client exited −2 on SIGINT; checkpoint recovery was accepted separately |

Sources and exact evidence: [FastTD3](FASTTD3_ORIGINAL_RESULTS_20261014.md), [Mulligan training and pilot evaluation](MULLIGAN_LOCAL_DIVL_20261014.md), [full-grid protocol](MULLIGAN_FULL_GRID_20261014.md), and [Real-Time EXPO](REALTIME_CATCHER_GPU_20261014.md). Each report records its code revision, data, settings, process status and artifacts.

The FastTD3 number is return, not success rate. The Mulligan and Catcher counts are task successes. The full-grid Mulligan cohort includes the earlier 50 starts; it is not an untouched final test set. No historical multi-seed paper aggregate or deployment-data scaling law has been reproduced.

## What the experiments establish

FastTD3 needs the environment's reward and a learned critic. More episodes cannot fix an incorrect action interface or a reward that carries no task information. Its original recipe uses large batches, many parallel worlds and a specific update schedule. Reducing it to one robot environment changes the learning regime. Our current author-task run did learn, especially late in the schedule, but it remains below the released curve. The log cannot assign the physics warnings to individual worlds or phases, so it cannot explain that gap.

Mulligan shows a complete original training path using released human-supervised deployment data. Its DIVL stage keeps the actor and normalizer fixed while updating the critic and value function. At deployment, critic ranking can change which of 32 actor samples is chosen. With one candidate, our IDQL and DIVL pilot outcomes are identical. With 32 candidates, the added stage fixes one pilot failure and loses three successes. That pilot does not show a gain from the added stage. Reusing the released data does not demonstrate a new human-feedback collection round.

Real-Time EXPO keeps slow base-policy inference and fast action editing as separate parts. It needs current observations, valid action prefixes, aligned rewards and replay, and measured delay. The Catcher run verifies that the released learner can update and produce complete frozen rollouts through this integration. Clean training-client completion was not accepted; the later frozen evaluation runs exited cleanly. Four reused starts cannot establish a reliable improvement. The available task adaptation also does not reproduce the missing exact EXPO simulation release configuration.

The [deployment-data study plan](DEPLOYMENT_RL_SCALING_PLAN.md) separates added data from added optimization. SGS, Mulligan collection and Real-Time EXPO address different decisions: reset selection, supervised collection, and delayed policy execution with learning. Combining them would require a new controlled experiment. The documented source check found no SGS author code release; no unofficial learner has been substituted.

## Implications for R1 Lite

The [R1 Lite transfer assessment](R1LITE_AUTHOR_RECIPE_TRANSFER_20261014.md) inspects the published Leela 0.5.0 interface. It uses 14 absolute arm/jaw targets at 15 Hz with a fixed base and torso. Its `step()` reward is always zero; a separate judge supplies outcomes. A verified judge-to-reward adapter is therefore required before task RL. Simulator-only object poses must remain explicit privileged inputs if an oracle state policy uses them.

The next transfer should use one published manipulation task. Verify native successful and failed replays, physical action scaling, reward, valid resets, and termination versus truncation. Then establish a frozen initialization and one bounded learning comparison on fresh starts. FastTD3 would learn a new state policy. Mulligan needs correctly labelled actor/critic data and an explicit supervision protocol. Real-Time EXPO needs measured inference delay and executed-prefix checks. Those are distinct integrations. This assessment supplies no R1 Lite training, whole-body control or hardware result.

## Architecture and team

The [architecture diagram](ORIGINAL_RL_RECIPE_RESULTS_20261014.mmd) shows the executed experiment boundaries and the proposed R1 Lite connection. The author programs own learning and task evaluation. The harness owns configuration, GPU0 allocation, finite run limits and process cleanup. Independent agents checked source identity, recipe assumptions and result presentation. Root owns execution, acceptance and publication. The UI shows accepted measurements and complete recordings.

The original-method campaign remains active until the full-grid result and its publication pass verification. The separate Taku laundry reconstruction is a visual experiment; it is not evidence of an RL policy or physical laundry handling. This report uses ASD-STE100 guidance. Full compliance was not checked.
