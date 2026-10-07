# Bounded ABC benchmark results

This campaign uses the released ABC bottles checkpoint in the native bimanual YAM simulator. The compute limit is one GPU for two hours. Exact run IDs, artifact checksums, runtime pins, and measured values are in `results_summary.json`.

The original three-episode pilot completed 3/3 tasks. The larger run stopped at its one-hour limit. It completed 43 of 50 planned episodes, with 35 successes. Seven planned episodes are excluded. Its published status is partial.

QF3 and ResFiT each completed four updates using actual policy observations and simulator transitions. These are small engineering adaptations. The QF3 adapter uses an immediate-return critic and a rank-four output adapter. ResFiT uses frozen visual and state features. Neither implements the full published training recipe.

Matched evaluation uses fresh environments, checked initial joint and object positions, identical seeds and noise streams, the same camera resolution, and eight executed actions per proposal. At the 1,000-step horizon, the frozen policy completed 0/3 tasks, QF3 completed 1/3, and ResFiT completed 0/3. This horizon limits each episode to 34 seconds of simulation. The 2,000-step follow-up completed 1/3 tasks for the frozen policy, 0/3 for QF3, and 1/3 for ResFiT. It was selected after the short results were observed; it is diagnostic, not an untouched final evaluation. Read the horizon with every success count. The paired ResFiT deployment adds an explicit 1e-4 interior codec limit. Its receipt reports clipped elements and physical deviations. This differs from the strict historical smoke at codec boundaries.

The paired initial-state flag checks a float32 `qpos` hash. It does not prove identity of every model property, velocity, or initial camera image. Fresh seeded CPU resets at both horizons reproduced the recorded positions and exposed model arrays. GPU camera observations and full action traces were not recorded. The same seed produced different closed-loop outcomes across runs, and the cause remains unresolved. The longer-horizon result must not be attributed to the extra horizon alone.

The first paired run failed at that boundary. Its failure receipt remains in the campaign. The corrected evaluator saves completed episode evidence before later failures.

Real-Time EXPO-FT has no measured training result. Its official JAX/OpenPI learner needs an ABC bridge, delayed replay, the learned noise-Q filter, and a current-state editor. R1 Lite and R1 Pro need verified controller, camera, reset, task, and policy contracts. Native YAM results do not transfer to either robot by renaming the embodiment.

At the matched cadence, eight control steps cover 272 ms of simulation. Compare whole-proposal latency with that interval. The simulator waits during policy inference; task success here does not prove a deployed real-time deadline. A deployed residual controller needs an action buffer and explicit observation-age, queue, and deadline measurements. Those measurements belong in the planned asynchronous EXPO-FT bridge.

The dashboard reads evidence. It cannot start jobs. It contains an architecture diagram and team roles. The coordinator controls budgets and measurements. The algorithm agent checks the objectives. The embodiment agent checks simulation and policy contracts. The independent reviewer checks code and claims. Jev supplies advisory criticism. This explanation uses ASD-STE100 guidance; full compliance has not been checked.

Total charged GPU campaign time: 6,463.41 seconds (107.72 minutes), within the 7,200-second limit. The remaining 736.59 seconds were left unused. All GPU jobs were sequential on GPU 0.
