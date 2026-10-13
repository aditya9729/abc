# FastTD3 reaching diagnostic

This stage tests a small physical skill before another five-bottle campaign.
The stage uses the original FastTD3 learner through its public environment factory.
It preserves the original networks, replay buffer, normalizers, and update functions.

The adapter keeps five bottles, 216 state values, 14 actions, and native MuJoCo physics.
Physics uses a 0.002-second step and 17 steps for each control.
The native five-bottle evaluator remains active at every control.
Its success value is separate from every reach result.
These native YAM experiments do not establish R1 Lite performance.

The learner reward is `exp(-d / 0.15)`.
Here, `d` is the smallest world-coordinate distance from either grasp site to any bottle centre of mass.
Distances use metres.
An owned forward-kinematics reader computes the distance from the actual native `qpos`.
The adapter checks that this reader preserves the full native `mjSTATE_INTEGRATION` state.
A separate owned MuJoCo data object checks initial bottle and grasp-site positions.
Every native warning count must stay zero.
The adapter also checks finite state values and the native control duration.

Every episode lasts 128 controls, or 4.352 seconds.
Reach requires five consecutive controls with distance at most 0.12 metres.
Reach does not end an episode.
Native five-bottle success also does not end this diagnostic episode.
The time limit produces truncation, with the actual terminal observation available for original FastTD3 bootstrap handling.
This termination choice is a diagnostic adaptation.
It prevents positive dense reward from encouraging delayed early termination.

The existing reset bank and carried arm behavior remain unchanged.
The live host measures reset distance before the first action.
An episode qualifies for reach metrics only if this initial distance exceeds 0.15 metres.
Already-near resets remain visible in the episode evidence.
They cannot establish qualified reaching.

Only three admitted TRAIN demonstrations determine the distance and duration settings.
Their first 0.12-metre crossings occur at controls 45, 61, and 62.
Their first five-control holds occur at controls 49, 65, and 66.
The 128-control duration exceeds twice the latest first crossing.
The five-control hold and 0.15-metre eligibility bound are declared diagnostic choices.
No VAL geometry determines these settings.
`TRAIN_SELECTION.json` pins the source cohort and exact original native replay traces.
Its first distance values follow control one; they are not reset geometry.

The separate runtime gate has 256 vector calls and two complete episodes per world.
The probe has at most 2048 vector calls.
A training progress monitor checks complete 256-call windows after 512 calls.
Each world must contribute a qualified completed episode before a window can affect stopping.
Three windows without a declared improvement can stop the probe naturally.
The earliest plateau stop is at control 1280.
The original current replay insertion and all scheduled updates finish before the loop exits.
The original final checkpoint export then runs.
The plateau test does not prove convergence or overtraining.

The learner uses 64 worlds instead of the author default of 128.
Batch size stays 32768 and replay capacity stays 51200 per world.
Aggregate replay capacity therefore changes from 6553600 to 3276800 transitions.
Critic updates per collected transition change from 0.015625 to 0.03125 after learning starts.
Learning starts, learning rates, critic support, AMP, compilation, and update math stay unchanged.
The checkpoint remains a policy export; it cannot provide an exact training continuation.

The new `abc_bench.fasttd3_reach_stage` has a separate canonical allowlist entry.
Existing campaign validators and the generic 7200-second parent limit remain unchanged.
The prepared job has a 3660-second parent bound.
The stage authenticates prior 64-world resource evidence before memory admission.
That prior evidence does not accept the new reward or episode behavior.
Long preparation requires a new, completed, canonical reach gate with the same source pins.
Root owns release, GPU 0 admission, native execution, and result publication.

A changing-policy TRAIN reach rate alone cannot isolate learning.
Reset layouts and exploration can also change that rate.
A matched initial and final deterministic actor comparison remains pending.
The original public update observer has no initial-actor export callback.
This candidate does not change original training code to add that callback.
It must not report an isolated learning improvement before that comparison exists.

The producing agent prepares code and CPU checks in isolated worktrees.
The root agent reviews the candidate after the active EXPO job ends.
The root agent releases the normal wheels and performs the native gate.
The root agent can then prepare the bounded probe from actual gate receipts.
Atlas, Engram, and Leela interfaces are outside this diagnostic.
No Jev or external reviewer runs in this candidate.

See [the architecture source](fasttd3_reach_diagnostic/ARCHITECTURE.mmd).
The build handoff lists exact commands, source pins, and executed evidence.
This text follows STE guidance. A full ASD-STE100 compliance check is unavailable.
