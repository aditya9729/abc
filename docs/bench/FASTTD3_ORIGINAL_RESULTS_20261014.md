# Original FastTD3: completed run and published recordings

The original 100,000-call run completed. The last periodic mean return is **572.72 at 95,000 calls**, versus **778.71** in the released curve at 12.16 million nominal steps. The local run learned substantially late: return rose from 82.01 at 55,000 calls to 572.72 at 95,000 calls. This completed result supersedes the earlier partial snapshot. A descriptive gap remains.

Root accepted run `author-subset-20261014T030724-7fc63f`. The original program exited 0 after 8385.47 seconds, about 2.33 hours. The independent source/data audit found no concrete mismatch. [Root acceptance](/home/user/aditya/RL/builds/fasttd3-full-acceptance-20261014/actual-full-01.json) and [published structured result](FASTTD3_ORIGINAL_RESULTS_20261014.json) contain the evidence and limits.

[Full comparison chart](/home/user/aditya/RL/builds/fasttd3-original-final-results-20261014/root-plot/full-reference.png) shows all 19 exact coordinate matches.

## Recipe and comparison

This is classic FastTD3 on `h1hand-hurdle-v0`, seed 1. It uses pristine author Fast `229ed59` and HumanoidBench `cb118903`. It retains 128 worlds, batch size 32,768, replay 51,200 per world, AMP/bf16, compilation, original rewards and all 61 resolved arguments. The completed original loop accounts for 12.8 million collected training transitions. There is no independent per-control trace.

All 19 original evaluation points are below. Match nominal `frame=global_step*128` exactly. No interpolation or best-point selection is used. At 95,000 calls, collection has already reached 12,160,128 training transitions; the original nominal coordinate is 12,160,000. Evaluation transitions are additional.

| Vector index | Nominal steps (M) | Local return | Released return | Difference |
| ---: | ---: | ---: | ---: | ---: |
| 5,000 | 0.64 | 7.44 | 6.49 | +0.95 |
| 10,000 | 1.28 | 7.41 | 10.59 | -3.18 |
| 15,000 | 1.92 | 8.84 | 18.61 | -9.77 |
| 20,000 | 2.56 | 13.14 | 29.66 | -16.52 |
| 25,000 | 3.20 | 25.83 | 49.01 | -23.18 |
| 30,000 | 3.84 | 33.20 | 85.59 | -52.39 |
| 35,000 | 4.48 | 48.52 | 115.56 | -67.03 |
| 40,000 | 5.12 | 60.65 | 173.20 | -112.55 |
| 45,000 | 5.76 | 91.71 | 218.64 | -126.94 |
| 50,000 | 6.40 | 89.50 | 319.63 | -230.13 |
| 55,000 | 7.04 | 82.01 | 377.23 | -295.22 |
| 60,000 | 7.68 | 181.15 | 496.74 | -315.59 |
| 65,000 | 8.32 | 215.25 | 490.29 | -275.04 |
| 70,000 | 8.96 | 279.08 | 610.66 | -331.58 |
| 75,000 | 9.60 | 364.93 | 658.83 | -293.90 |
| 80,000 | 10.24 | 443.09 | 689.10 | -246.01 |
| 85,000 | 10.88 | 467.92 | 710.21 | -242.29 |
| 90,000 | 11.52 | 565.25 | 735.35 | -170.10 |
| 95,000 | 12.16 | 572.72 | 778.71 | -205.98 |

The final saved weights have index 100,000. **The original default provides no 100,000-call evaluation.** The latest measured return belongs to the 95,000 evaluation. It is a 128-world mean first-episode return. It is not a task-success rate. The released aggregate reaches 19.84 million nominal steps and lacks per-seed configs, code revision and runtime metadata. This single current-release seed does not reproduce the historical three-seed experiment or isolate the gap's cause.

## State, history and physics

The closed history has 999 finite scalar rows and 19 evaluations. Every scalar row matches the decoded original log. Root's pinned CPU checkpoint reader loaded finite saved tensors. The saved index is 100000. The observed normalizer counts match the source expectation. Source control flow implies 199,978 critic and 99,989 actor updates. **Optimizer counters are not saved or observed.** The checkpoint omits optimizer, replay, RNG and world state, so it cannot establish exact continuation.

The complete closed stdout contains **nine printed QACC warnings**. MuJoCo 3.1.6 resets native physics state on this condition. Warning suppression makes the printed count a lower bound. The combined log identifies neither affected worlds nor train/eval/render phases. It has no Python traceback marker. Finite history and tensors do not establish warning-free physics. No recovery effect or compilation cause has been isolated.

The independent reviewer authenticated the acceptance inputs, final checkpoint and 19 GIF byte pins. The reviewer rechecked 25 Fast and 2,977 HumanoidBench tracked files, 12 and 2,780 normal installed payloads, and all 9 recorded runtime versions. The old canonical wheel matches its four accepted module hashes. Root released new canonical code afterward. The reviewer did not rerun the sealed checker or require current code to retain historical hashes.

## Media and handoff

Three preselected original recordings are published at [the results viewer](http://127.0.0.1:8770/original-methods/#fast-original-heading). They correspond to 5,000, 50,000 and 95,000 calls. The MP4s preserve all 36, 462 and 500 source frames, respectively. Their durations remain 1.08, 13.86 and 15 seconds. These are existing one-world render recordings. They are separate from the 128-world evaluations and the final 100,000-call checkpoint. No new rollout was run for publication.

Root's full CPU decode and browser checks passed. The browser checked actual HTTP bytes and range requests, all 19 return points, playback, seek and refresh. It preserved the older panels. The assessor authenticated the receipts and media bytes, compared the previous snapshot, and viewed the saved desktop, mobile and late contact images. Only the new `fasttd3_original` record and `snapshot_utc` changed. [PUBLICATION_REVIEW.json](/home/user/aditya/RL/builds/fasttd3-original-final-results-20261014/PUBLICATION_REVIEW.json) records the checks and their limits. A byte-identical reviewed snapshot is archived for later viewer updates.

Root owns execution and publication. The assessor owns this additive review. The live browser and decoder were not rerun by the assessor. No model, native environment, numerical-library import or GPU job was executed for this review. No shared code or runtime was changed. The original report and seal remain intact. Video timing does not establish wall-clock control speed or supply the missing per-control native trace.

[EVIDENCE.json](/home/user/aditya/RL/builds/fasttd3-original-final-results-20261014/EVIDENCE.json) pins the external inputs. [ALL_19_POINTS.csv](/home/user/aditya/RL/builds/fasttd3-original-final-results-20261014/ALL_19_POINTS.csv) preserves exact values. [ARCHITECTURE_PUBLISHED.mmd](FASTTD3_ORIGINAL_PROGRAM.mmd) shows the workflow. This explanation uses ASD-STE100 guidance. It has not had a full compliance check.
