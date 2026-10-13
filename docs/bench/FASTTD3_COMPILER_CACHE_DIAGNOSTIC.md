# Full-task compiler-cache resource diagnostic

This candidate compares the public MuJoCo compiler asset cache at its default
capacity and at zero capacity. It uses the original full five-bottle native
`RewardWorld`. It does not train a policy or change the full 64 campaign.

The module is `abc_bench.fasttd3_compiler_cache_diagnostic`. The only change to
an existing source file is its named entry in the canonical subset allowlist.
All original simulator, host, learner, replay, update and production memory
admission code stays exact. The new module is separately packaged and pinned.

The canonical stage checks the live lease before filesystem changes or child
launch. It authenticates the old completed 64 gate only for resource evidence,
using the old executed source pins. Those pins cannot admit current training.
The stage runs four cases sequentially: TRAIN ranks 0 and55, each with default
and disabled compiler caches. At most one child lives at a time. Each case is a
fresh process. It starts with CUDA hidden, graphics disabled and one CPU thread.
An internal worker requires its live immediate stage parent, matching process
start identity, process group and config hash before any native import/cache
call. Every child remains in the canonical owned process group.

Each case calls the public cache capacity setter before the first native model
is constructed. It requires MuJoCo 3.8.1, an empty initial cache and the published
500 MiB default. Default mode changes nothing; disabled mode sets capacity 0.
No private scene-cache field, allocator, native symbol or cgroup is changed.

The unchanged full 64 host configuration keeps the seed bank, task, 3540 horizon,
216-value state,14 actions, physics and full reward. Only ranks 0 and55 are selected for
this resource test. This is not a reduced-world training recipe or a64-world
qualification. Each case requests13 successive original TRAIN resets and issues
two float32-normalized zero actions per reset through original `RewardWorld.step`.
The original reset carry/fallback logic remains in use. The comparison requires
13 distinct compiled-scene keys from the recorded randomization metadata. The
unchanged scene LRU has 12 entries; the default preparation adds its own scene.
There is no direct private-cache occupancy inspection.

The read-only wrapper around the instance's public `env.step` calls the bound
original exactly once. It captures the returned authoritative native judge,
physics and full post-control INTEGRATION before `RewardWorld.step` can autoreset.
It returns the identical tuple unchanged. If the original fails, the wrapper
does not retry it or fabricate a return. Initial judge observation uses a separate
public original evaluator on copied qpos; the temporary evaluator is released
before the next reset. A terminal control preserves its actual post-control
evidence but fails this declared short lifecycle probe rather than silently
changing its reset sequence.

Each initial/post-control row records complete raw float64 INTEGRATION bytes,
model-array hashes, numeric physics options/flags, randomization and reset
attempts, 216-value state, native controls and the full native judge. Control rows also
record original reward/action output. The observer checks that it did not change
INTEGRATION. All seven native warning counts must be zero. The parent authenticates
and compares the actual full state/XML bytes and every native content field.
Paths and measured resource timing are excluded from native equivalence.

Resource samples record current RSS/PSS, process lifetime peak RSS, public cache
capacity/occupied bytes, elapsed time and exposed model/data buffer sizes. Buffer
sizes are not a decomposition of total native memory. Per-case peak differences
are observations from separate processes. Their sum is not an observed aggregate
peak. No receipt declares full 64 RAM sufficient.

The one-child preflight floor is explicitly **unvalidated**: 13 times the authentic
historical maximum worker RSS, plus 25 percent and 8 GiB. It currently evaluates to
70,606,218 KiB. This new diagnostic-only estimate is not a hard memory cap. It does
not modify or replace production 64/128 memory admission. Dynamic system pressure
and future reset allocations remain uncertain. The parent checks it before each
case. This test cannot certify a long 64-world run, even if native bytes match.

Watchdogs:900 seconds per case,30 seconds owned cleanup,3900 seconds stage and
4200 seconds canonical job. The next case starts only after the prior child is
reaped. Failure, timeout, cancellation, missing receipts, warnings, changed bytes
or forced kill prevent a completed compatible result. Partial case data/logs are
preserved; cleanup touches only the owned child. The canonical coordinator owns
final process-group cleanup.

Receipts are separately named `resource_equivalence_only` and
`native_case_resource_only`. They are not trainers, checkpoints or qualifications.
`training_producer=false`, `qualification=false`, benchmark score unavailable and
full 64 memory sufficiency unavailable remain explicit. Original `RewardWorld.close`
also writes its historical reward-summary file. Its episode-index counter includes
these manual resets; it is not a completed-task count for this diagnostic. Never
score that file or this stage as policy learning.

Root owns normal release, canonical admission, actual native execution and the
decision after results. The preparation agent owns the isolated code, installed
CPU tests and sealed handoff. CPU process/cache/state doubles check boundaries,
not native equivalence or memory savings. No native case ran in preparation.
[Architecture](FASTTD3_COMPILER_CACHE_DIAGNOSTIC.mmd). Text follows ASD-STE100
guidance; formal compliance was not checked.
