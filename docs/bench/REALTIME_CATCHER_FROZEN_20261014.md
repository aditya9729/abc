# Frozen Catcher development comparison

RTC-guided BC24 succeeds on 1/4 starts. The short learned R2 policy succeeds on 0/4.
These are eight accepted native CPU episodes on one original 2D Catcher level.
This small development cohort does not establish reliable improvement or paper performance.

| Seed | RTC BC24 success / return / controls | Learned R2 success / return / controls |
| --- | --- | --- |
| 100001 | success / 1 / 75 | failure / 0 / 68 |
| 100002 | failure / 0 / 59 | failure / 0 / 71 |
| 100003 | failure / -1 / 62 | failure / -1 / 69 |
| 100004 | failure / -1 / 65 | failure / -1 / 64 |

Both arms use matched native second-reset states, delay 4, C4/H8, ten Euler steps, and action noise 0.1.
RTC BC24 uses the original critic-free RTC sampler. R2 uses the learned whole policy with selection and editing.
The R2 training check saved 34 base/editor/temperature and 680 critic/filter/encoder updates.
This evaluation makes zero learner updates. Parameters, optimizers, clocks and policy files remain unchanged.
Original learner equations and policy samplers remain intact. Catcher state/backend/task hooks are declared adaptations.
The evaluator adds a Catcher-only post-fill scheduling recheck and read-only chronology observations.
The backend adds the exact original padding method required by RTC.
The initial missing-interface attempt and the later process-cleanup failure remain unavailable, each with denominator zero.
The final seed uses its explicit fresh operational retry. No failed producer is normalized into a task outcome.
Eight positive-control videos fully decode. Playback uses 30 simulation frames per second.
8/8 native reset-to-terminal intervals exceed their simulation duration.
Playback does not prove wall-clock real-time control. Submit/consume records do not prove direct solver overlap.

The installed dependency report records four metadata conflicts. Earlier two-conflict statements are superseded:

- Kinetix requires JAX 0.4.35; JAX 0.5.3 is installed.
- Catcher task requires ml-collections 1.1.0; 1.0.0 is installed.
- The historical EXPO adapter requires expo-ft 0.1.0; 0.1.1 is installed.
- jaxued requires Orbax 0.5.3; 0.11.13 is installed.

Root owns acceptance and publication. Author code owns policy sampling and evaluation.
The task agent owns task/transport boundaries. The publication agent owns the publication helpers.
See the [evidence pins](REALTIME_CATCHER_FROZEN_20261014.json) and [architecture](REALTIME_CATCHER_FROZEN_20261014.mmd).
STE guidance used. Full ASD-STE100 compliance was not checked.


[Open the full comparison](http://127.0.0.1:8770/original-methods/#catcher-frozen-heading).
The actual browser passed playback, seeking, refresh, equal column widths, and desktop/mobile checks.
HTTP full and byte-range responses match all eight video hashes. Prior result sections stay unchanged.
Five DOM checks pass. The normally installed full wheel has 1,149 authenticated payloads, including 914 assets.
The historical full-suite failures remain unresolved; this milestone does not claim a passing full suite.

The [native online report](REALTIME_CATCHER_ONLINE_20261014.md) defines the short R2 producer and its task adaptations.
This checkpoint follows a 3,072-step integration schedule, not the paper's full 100,000-step, four-seed experiment.
The [scaling plan](DEPLOYMENT_RL_SCALING_PLAN.md) keeps collection, training, and independent evaluation separate.
Four fixed development starts cannot establish a scaling law or justify an R1 Lite performance claim.
