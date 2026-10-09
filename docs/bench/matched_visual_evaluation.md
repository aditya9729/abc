# Matched visual evaluator coordinator

The isolated ABC package adds `visual-method-evaluation` to the common runner.
The route supports six declared ABC/YAM protocol names in NRH 0.4.13, evaluator revision 2.
It sends one worker through `execute_command`. It keeps the existing training routes unchanged.

| Protocol | Learned export needed |
| --- | --- |
| `resfit-frozen-nominal` | No |
| `resfit-learned-mean` | Yes |
| `expo-frozen-async-n1` | No |
| `expo-full-learned-n8e8` | Yes |
| `qf3-frozen-h30c15-nortc` | No |
| `qf3-learned-online-h30c15-nortc` | Yes |

These algorithm ports have different computation costs and declared training recipes.
They do not establish paper reproduction, real-time performance, or R1 Lite transfer.
Software fixtures do not establish checkpoint eligibility or native performance.

## Public route and trust

The following is a future launch template. It has not been executed against the native campaign.
Root must admit the isolated coordinator, final worker package, resources, and exact request first.

```sh
ABC_BENCH_REPO=/home/user/aditya/RL/abc \
  /ABSOLUTE/ISOLATED/COORDINATOR/bin/python -I -B -m abc_bench.runner \
  --algorithm visual-method-evaluation \
  --evaluation-request /ABSOLUTE/ROOT_REQUEST.json \
  --evaluation-request-sha256 REQUEST_SHA256 \
  --evaluation-request-bytes REQUEST_BYTES \
  --results /home/user/aditya/RL/abc/outputs/bench \
  --timeout-seconds FINITE_PARENT_SECONDS
```

The three new request options belong to the ABC runner. They are not NRH options.
Legacy checkpoint, training, task, seed, video, state, and resume overrides are refused for this route.
The worker CLI remains `python -I -B -m nrh.visual_method_evaluation --config --out --campaign-directory`.
The route adds the existing `--trusted-native-admission` path/hash/bytes options.
Learned protocols also receive the existing `--trusted-checkpoint-admission` path/hash/bytes options.
The protocol, export, video selection, and resume pins remain in the pinned configuration.

`root_abc_matched_visual_evaluation_request`, schema 1, has these exact fields:

| Field | Contract |
| --- | --- |
| `config` | Absolute path, byte count, SHA256 for the unchanged NRH configuration |
| `identity` | Exact resolved revision-2 protocol identity; protocol ID is its canonical JSON digest |
| `worker` | Literal isolated venv Python pin, resolved binary, prefix/version, pyvenv config, own NRH sources, METADATA/RECORD, plain `.pth` and selected dependency metadata pins |
| `software_binding` | Root release binding for NRH 0.4.13; accepted source/wheel/install report pin and every installed NRH source hash |
| `native_admission` | Existing root NRH native admission plus coordinator source, worker, and resource bindings |
| `checkpoint_admission` | Separate complete-checkpoint trust pin for a learned protocol; null for a frozen protocol |
| `coordinator_sources` | Exact hashes of the actual installed runner, dispatch, and bootstrap |
| `campaign_directory` | Literal original absolute `/home/user/aditya/RL/abc/outputs/bench` |
| `cleanup_s` | Finite positive cleanup allowance within the parent deadline |
| `receipt_max_bytes` | Positive root-admitted byte bound, at most 512 MiB |
| `native_resources` | Resource admission pin, host resident limit, GPU memory budget, and explicit optional virtual address limit |
| `schema_version`, `kind` | Exact schema and request kind |

Every pin has exactly `path`, `bytes`, and `sha256`. No mutable `latest` alias grants trust.
`root_abc_visual_worker_release_binding` must have status `accepted_cpu_source_wheel_install` and version `0.4.13`.
Its `nrh_sources` map must match all installed Python sources. Its `source_admission` is separately pinned.
The native admission must bind `worker_binding`, `coordinator_sources`, and `native_resources` to this request.
These coordinator-only fields extend the root certificate. They do not change the accepted NRH CLI or lease rule.
The worker still validates native capture sources, loaded origins, construction identity, and complete-checkpoint trust.

The route requires exactly 50 development episodes for Bottles, with unique ordered IDs and seeds.
Final cohorts are unsupported. Each episode has at most 1000 returned controls.
At most three video IDs are selected before execution.
Actual cohort, checkpoint, model, and video artifacts were not used in coordinator software tests.

## Lease and lifetime

The route opens the existing original lock and ledger. It never creates a worktree budget.
It keeps prior charges and refuses active or unresolved campaign ownership.
The existing GPU occupancy check occurs only after request validation and lock acquisition.
The finite allocation includes execution, receipt validation, and cleanup.

The parent reserves the complete allocation before one worker dispatch.
The lease records a random nonce, boot ID, parent PID/start ticks/group, and original lock inode.
The child starts a new process group. A stdlib bootstrap verifies the exact direct parent and inherited lock.
The bootstrap forks a guardian outside the evaluator group. The guardian retains the same inherited flock.
The parent records worker and guardian lifetime tokens, then publishes its acknowledgment.
The bootstrap verifies that acknowledgment and original ledger before direct `exec` of the unchanged NRH CLI.
The evaluator therefore keeps the ABC runner as its direct parent.

The guardian checks only owned process identities and their bounded child lists.
It does not scan all system processes, query a GPU, or control unrelated jobs.
PID reuse, start-tick changes, boot changes, group changes, and nonce mismatches deny ownership.
The parent keeps the leader unreaped until its group is terminated, then reaps its owned descendants.
Normal completion, cancellation, and timeout stop a TERM-ignoring owned descendant.
The guardian holds the lock through receipt validation and group cleanup.
An immutable stop/outcome handshake precedes normal release.
If the parent dies after guardian release but before settlement, the full active record still blocks another launch.
The clean guardian outcome then proves group cleanup, but no guardian remains to change that record's status.
Root must reconcile this terminal crash window. A missing final receipt grants no completed score.

Hard parent death leaves the full reservation and explicit unresolved lease evidence.
The guardian kills only a verified member's original group and waits within the admitted cleanup deadline.
If cleanup or ownership cannot be proved, the lease stays unresolved. Root must reconcile it before another job.
There is no automatic refund, score promotion, stale-lock reset, or ledger migration.
The guardian cannot survive its own SIGKILL or host failure. Such events require external root recovery.
Uninterruptible kernel tasks can prevent proved cleanup; the unresolved tombstone remains the admission gate.

The controller requires the main Python thread, one Python thread, and one kernel thread.
It refuses a threaded caller before it changes shared signal, timer, or subreaper state.
TERM, INT, and ALRM are retained during child birth, identity binding, and bounded cleanup.
Normal wait checkpoints raise retained cancellation without extending absolute deadlines.
Handler installation and restoration use a temporary signal mask.
Partial startup also restores caller handlers, mask, timer schedule, and subreaper state.
An expired one-shot timer is not rearmed.
Standard signals can coalesce; these records do not count every signal delivery.
Each scope records original exec dispositions separately from cancellation callbacks.
Nested probe and dispatch children inherit original SIG_IGN and the full caller mask.
Only ignored dispositions need a child pre-exec callback. The controller is single-threaded.
Originally blocked handlers stay untouched. Their pending bits remain caller-owned.
The scope never drains or directly delivers those caller-blocked signals.

Terminal receipt publication remains inside retained cancellation ownership.
Cancellation during actual-wall settlement, tentative publication, or handler restoration revokes accepted complete-control credit.
The raw worker receipt remains unchanged. A late-cancellation receipt also retains its producer summary.
A failed replacement publication cannot prove a valid completed invocation; root must inspect the actual process result.
The terminal callback checks the original absolute deadline before and after tentative success publication.
The last owned pending snapshot after publication and handler restoration is the explicit cancellation cutoff.
Signals after that snapshot follow restored caller semantics, including pending caller-blocked bits.
This cutoff precedes the final mask restore. It does not promise immunity from later cancellation or hard death.

Before reservation, a fixed `python -I -S -B -c` probe reads isolated interpreter metadata.
Site is disabled for that probe. It imports no NRH or test dependency.
The probe refuses an expired absolute deadline before child birth, including expiry during setup.
It uses protected ownership assignment, finite cleanup, and a 32 KiB output bound.
Its output hash, paths, interpreter identity, and closed lifetime enter bootstrap provenance.
Its CPU and wall cost consume the original aggregate deadline before lease arming.
The parent resolves effective startup modules across stdlib, worker site, and admitted plain `.pth` paths.
Importable `sitecustomize` and `usercustomize` modules, packages, bytecode, and extensions are forbidden.
The exact NRH package root and evaluator entry origin must match the source pins.
Runtime filesystem immutability and final repaired NRH release admission remain root duties.

## Receipt and resource accounting

| Worker result | Coordinator evidence |
| --- | --- |
| Exit 0, healthy completed 50-row receipt | Completed producer evidence; independent reader/root performance admission still pending |
| Exit 0, clean incomplete receipt | Incomplete evidence; worker exit 0 is preserved and runner CLI returns 1 |
| Nonzero exit, typed failed receipt | Failed evidence and reconciled physical counts; no completed score |
| Exit/status mismatch, invalid identity/counts, late preservation or cleanup error | Failed evidence; no completed score |
| Guardian failure, watchdog, hard parent death, unproved cleanup | Full reservation retained, unresolved ownership, no score/refund |

Physical counts cover the current invocation. Completed resume rows supply no new physical controls.
Known returned controls, captured/judged controls, uncaptured returns, and unknown attempts remain separate.
Accepted complete controls and discarded partial controls remain separate.
`reported_complete_row_control_steps` preserves physical controls from reported complete rows.
`accepted_complete_control_steps` is zero after any failed invocation or parent cleanup failure.
History placement success and native current-placement success remain separate in producer metrics.
Generic dashboard success, episode, and latency fields remain unavailable pending independent admission.
Healthy resume requires an exact ordered completed prefix from healthy completed/incomplete invocations.
Failed invocations cannot supply resume credit, even if their rows look complete.

The reader checks size before reading, rejects duplicate/nonfinite JSON, and validates typed counts and vectors.
QF3 can retain large proposal arrays and command records. A legitimate 50-episode receipt can exceed 64 MiB.
Root therefore admits an explicit finite receipt bound up to 512 MiB.
Hashing, JSON decoding, validation, and resumed-prefix checks consume parent time and host memory.
Each host sample has exact `iteration`, `returned_control_index`, `selection_dispatch`, `step`, `capture`, and `whole_loop` fields.
Indices are ordered exact integers. Durations are finite nonnegative numbers, and booleans are forbidden.
All returned controls retain a sample. At most one additional failed loop can follow them.
Only partial step/capture phases can be null, with physical-return causality checked.
The phase sum cannot exceed the enclosing loop beyond a 10 ns rounding allowance.
These checks validate retained schema and causality; they do not authenticate native timing measurements.
The watchdog covers that work. Large native receipt costs still need root profiling.

`root_abc_visual_native_resource_admission` binds status `accepted_for_serial_native_development_evaluation` and exact `limits`.
The limits are `host_rss_limit_bytes`, `gpu_memory_limit_bytes`, and `virtual_address_limit_bytes`.
The guardian samples resident host bytes for the parent, guardian, worker, and discovered owned descendants.
An exceeded host limit requests cancellation. GPU memory is a declared root budget; this software adds no GPU query.
An explicit virtual-address limit is applied before NRH exec. Null inherits the parent operating-system limit.
The bootstrap records inherited virtual-address and CPU limits for root inspection.
CUDA virtual reservations can greatly exceed resident memory. An address-space limit is not a physical-memory allocation.
The 16 GiB address-space cap applies to CPU software tests. It is not a native worker default.
Native bounds, actual parent inherited limits, resident/GPU profiling, and model eligibility require later root admission.

## Verification and roles

Focused tests use a synthetic stdlib NRH package, fake trust metadata, and owned fake original campaign directories.
The synthetic construction is explicitly ineligible for the real native evaluator.
Tests cover all six routes, exact CLI/direct parent, byte/RECORD/source pins, exit/status mapping, resume counts, and lifetime faults.
An isolated installed ABC wheel must pass the same focused fixture tests and import-origin checks.
The unchanged training dispatches have separate CPU regression coverage.

Read [the architecture](matched_visual_evaluation.mmd) and [the STE handoff](matched_visual_evaluation_ste.md).
The coordinator owner writes code and CPU evidence. The NRH owner defines the frozen public worker interface.
An independent reviewer verifies the frozen result. Root admits software, runtime trust, native resources, and later launches.
Manual OpenQodex review and publication are separate root gates. This implementation performs neither.


## Additive second repair

This repair changes cancellation ownership and evidence publication only.
It preserves the previous 233 coordinator contracts and the legacy training/dispatch bodies.
The first repair's source, wheel, tests, peer findings, and fixtures remain immutable.
Owner regressions extend the sealed peer witnesses to all seven nonempty TERM/INT/ALRM sets.
They cover actual nested child exec, blocked pending bits with SIG_IGN, deadline refusal, and terminal cancellation.
Additional cases separate post-cutoff caller delivery from owned cancellation and verify publication error recovery.
These are owner tests. Fresh independent source and installed-package review remains required.
Actual NRH, native execution, models, datasets, GPU, and performance admission remain separate root gates.
