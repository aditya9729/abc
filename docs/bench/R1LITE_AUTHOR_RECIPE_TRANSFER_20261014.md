# Transfer the author recipes to R1 Lite

This is a transfer assessment, not an R1 Lite learning result. Keep the original
runs and their task scores separate from a future R1 Lite experiment. The target
is R1 Lite with a fixed base and torso for the first manipulation task.

## The available public interface

Root inspected the published Leela v1 manifest, `leela_sim` 0.5.0 wheel, and
demonstration index. Their hashes match the 8 October handoff. This release
supports a bounded simulation smoke. The newer cans handoff still awaits Leela
acceptance; neither packet establishes real-robot training admission.

The wheel, rather than newer documentation for 0.6/0.7, defines this boundary:

It requires MuJoCo 3.13.x. Keep this task runtime separate from the original
FastTD3 Hurdle runtime, which uses MuJoCo 3.1.6.

| Boundary | Published 0.5.0 behavior | Required consumer choice |
| --- | --- | --- |
| Action | 14 absolute targets: left six arm angles, jaw width, right six angles, jaw width; radians and metres | Define the learner-to-physical transform and save proposed and applied actions. The simulator reports clipping. |
| Control | 15 Hz, with 40 physics steps at 600 Hz; position servos | Keep these rates fixed for the first port and measure inference time against this deadline. |
| Reward | `step()` always returns 0.0; the separate judge supplies success/failure | Define and verify a task reward adapter before training. Feeding the raw scalar into RL supplies no task reward. |
| Observations | Joint state, jaw widths, TCP poses, torso, optional RGB, and privileged object poses | Keep simulator-only object truth out of a hardware-facing actor, or label the actor as an oracle simulation baseline. |
| Reset | Default scene, optional object perturbation and `init_arm_q`; jaws open; contact and settle checks | Use only validated training resets. This interface does not expose the full grasped-state reset bank used by SGS manipulation. |
| Episode end | Success held for the required interval or dropped object terminates; time limit truncates | Preserve these two flags and the selected author's bootstrap rules. |

The environment does not reset itself or retain a sticky end flag. The collector
must stop and reset after an episode ends; further `step()` calls keep simulating.

The index has 489 trajectories, of which 173 carry the producer's `robust` flag.
All 173 are one-object tasks: 164 `only_first` and nine `only_second`. They are
concentrated in the episode-1 family: 156 rows, versus nine for episode 0 and
eight for episode 2. These cousins are not 173 independent real demonstrations.
Group related source episodes and scene families when making splits.

The index records eight robustness trials for 30 of those rows and sixteen for
143. This qualifies the board's earlier blanket sixteen-trial description.
These are producer metadata counts, not a new replay or a check that every trial
was independent. Root authenticated only the listed published files, not all
21,690 manifest members or every trajectory.

## What changes for each recipe

**FastTD3:** retain the original learner and make the environment change explicit.
The Hurdle actor's observation size and 61 actuator outputs do not transfer to
the 14-target R1 interface. A new state policy starts from scratch. Define its
observation, action normalization, and task reward. Sparse placement success may
not supply enough early exploration; test this rather than adding unreported
reward terms. Comparing this state policy with a pretrained VLA measures
different training strategies.

**Mulligan:** retain the original actor/critic data rules and collection records.
Leela's generated trajectory cousins are synthetic demonstrations. They are not
new live human interventions. Do not relabel them as operator corrections to
satisfy a loader. Using them for initialization requires an explicit data
adaptation. A new supervised deployment round also needs an operator protocol
and measured intervention effort.

**Real-Time EXPO-FT:** retain the released learner, prefix conditioning, action
sampler, editor, critics, and replay/update rules. Supply the task through its
environment/backend interfaces. Four controls at 15 Hz take about 267 ms; at
30 Hz they take about 133 ms. Measure R1 inference delay before selecting the
control profile. Verify committed prefixes against executed commands, including
the physical action transform. Simulation-clock delay alone does not prove that
execution meets a wall-clock deadline.

**SGS:** use outcomes to choose a validated reset configuration. The public
0.5.0 reset API can vary starts but cannot restore an arbitrary object-in-gripper
state. A faithful manipulation reset bank needs a further published Leela
interface and physical checks. The SGS author code is also still unavailable
at this review. Do not call a locally designed reset sampler an author result.

## The first transfer experiment

After the original-task results are accepted, choose one published one-object
task. Recheck its bundle and successful/failed native replays through the public
wheel. Verify reward, action transform, observations, and episode endings.
Then establish a frozen initialization and one bounded learning comparison on
fresh simulation trials. Keep base/torso motion, new policy families, and whole
body control as later changes. A static torso reset is not whole body control.

The artifact inspection is
`builds/r1lite-author-transfer-review-20261014/INSPECTION.json`, with the
reproducible, standard-library-only reader beside it. No model, simulator, GPU,
installation, or hardware ran in that inspection.

Root owns the learner adapter and result interpretation. Leela owns scenes,
physics and reset interfaces. Engram and Atlas own their published source and
annotation decisions. The [diagram](R1LITE_AUTHOR_RECIPE_TRANSFER_20261014.mmd)
shows the proposed connection. STE guidance was used; full compliance was not
checked.
