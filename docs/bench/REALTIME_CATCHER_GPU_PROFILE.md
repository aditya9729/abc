# Catcher GPU learner and CPU simulator profile

This profile adds process routing to the existing original-program launcher.
It does not change the author learner, replay schedule, control loop, or model.

Use `abc_bench.author_program` schema2 with
`task_profile="kinetix_catcher_gpu_learner_cpu_env_v1"` and
`program="train_pi_robo.py"`. A client is required. Both processes use the same
normally installed Catcher runtime.

The server `--config` and `--config_task` must name that runtime's installed
`realtime_expo_catcher_task/model_config.py` and `config.py`. The client must use
`--config-task-path realtime_expo_catcher_task.config`. A bounded stdlib metadata
process checks those files against the normal distribution RECORD. It imports
no task, learner, JAX, or simulator. Nested config overrides are rejected because
they could change this task routing. Unknown role fields and CUDA environment
overrides are also rejected.

Schema2 requires absolute worker, model, and task configuration paths. Validation
and execution must select the same files across their different working directories.
Client task-selector option names normalize Tyro's underscore/hyphen aliases.
Exactly one selector is required. Its module value remains unchanged.

Start through the canonical runner with inherited `JAX_PLATFORMS=cuda`.
The worker retains the exact GPU UUID that its existing lease check authenticated.
Its server fixes legacy `JAX_PLATFORM_NAME=cuda`, including when the shell has a
stale CPU value. Only the client gets hidden CUDA and explicit CPU JAX settings.
The receipt records both role environments. The parent environment is unchanged.
Schema1 retains its existing shared environment and admission behavior.

The canonical Catcher receipt uses task `Catcher-v3` and embodiment
`kinetix_symbolic`. Caller-supplied conflicting task metadata is rejected.
Both the job limit and effective runner timeout must contain the worker limit
plus **100 seconds** for cleanup. Existing limits remain worker 7,100s and
parent 7,200s. A proposed worker 3,600s / parent 3,700s pair fits those limits.

Use a fresh output and trace directory. Preserve the original model, dataset,
update, and timing flags. The launcher appends its owned output path. Use offline
W&B and workspace cache paths. Disable persistent JAX cache writes explicitly
when using the existing author helper that selects a home-directory cache.

The worker owns its direct children. The canonical coordinator owns the complete
process group and GPU lease. A zero process exit is not checkpoint acceptance.
Root must inspect native closure, full checkpoint restoration, replay retention,
actual clocks, timing, memory, group cleanup, and source pins separately.

The first proposal uses all complete expert episodes, B64/UTD20, and the original
transition-derived episode update schedule. It is a timing adaptation. It does
not supply the unreleased Catcher simulation recipe or a frozen policy score.
Metadata checks find matching JAX/CUDA12 packages, with four disclosed package
conflicts. Actual GPU execution remains a separate qualification.

The development checks use explicit CPU process doubles. They check routing,
cleanup, legacy compatibility, task identity, overrides, and watchdog containment.
They do not load a learner or run physics. Root owns normal installation and
native execution after the current Fast run releases GPU0.

[Architecture](REALTIME_CATCHER_GPU_PROFILE.mmd). The author code owns learning.
This agent supplies the process boundary. Root owns release and execution.
STE guidance used; full ASD-STE100 compliance was not checked.
