# Original Mulligan evaluation video

The new `abc_bench.mulligan_video` module records one to three selected
Square-Narrow starts. It calls the installed author's `evaluate_manifest` and
`step_rollouts_from_obs`. It does not implement a policy, learner, controller,
reset, action transform, or evaluation loop. The author source remains unchanged.

The module temporarily substitutes an observer subclass for the public
`SyncVectorEnv`. Each reset and step calls the original method once and returns
the original result. Selected vector slots call the original environment factory
with only `use_render_wrapper=True` in place of its explicit `False`. This enables
the original offscreen `agentview` renderer. All physics, observation, controller,
reward and reset options remain the author's options. Both substitutions are
restored in `finally`. This path requires a dedicated single-process child and
the author's explicit `--use-sync-envs` option; it does not claim asynchronous
timing equivalence.

The recorder streams uint8 RGB frames at 240×320 and 20 FPS to FFmpeg. It keeps
one initial frame and one frame after each counted native control. A worker's
video ends at its first success, `done`, or horizon, even if the author continues
to step that worker while other batch members finish. The MP4 uses simulation
time. Its duration is not a measured wall-clock control rate.

The output directory must be fresh. Resume, shards, and more than three video
points reject. Existing policy results and videos are never overwritten. A
recording requires `MUJOCO_GL=osmesa`; root supplies the task-local Mesa library
path. There is no GPU renderer fallback.

For each selected point, `videos/receipt.json` binds the video and trace hashes,
manifest pin, complete point metadata and authoritative author outcome row.
Acceptance requires the observed success, reward and length to equal that row,
the MP4 to be nonempty, and the original evaluator to return after clean close.
Rendering, encoder, outcome mismatch or native cleanup errors propagate and leave
a failed receipt. Completed author rows can remain available after cleanup fails;
they do not turn that job into a clean success. A forcibly terminated process
without a completed receipt is not accepted.

Each trace contains hashes of unchanged raw action arrays and qpos/qvel/simulation
time before and after each counted control. Rendering also checks those states,
Python/NumPy global RNGs and the environment generator for immediate changes.
These are diagnostic state hashes, not a full MuJoCo integration-state proof.
`--trace-only` produces the same selected traces with rendering disabled. Matched
manifest, batch size, original evaluation seed and reset seed permit a native
render-on/off comparison; seeds alone do not guarantee deterministic CUDA kernels.

After normal installation in the root-owned original dependency prefix, the
child command is:

```sh
MUJOCO_GL=osmesa \
LD_LIBRARY_PATH=/home/user/aditya/runtimes/mulligan-osmesa/root/usr/lib/x86_64-linux-gnu \
<original-child-python> -I -u -B -m abc_bench.mulligan_video \
  --artifact-path <authenticated-local-author-checkpoint> \
  --point-manifest <same-authenticated-50-start-manifest> \
  --output-dir <new-output-directory> \
  --use-sync-envs --num-envs 10 --max-steps 400 \
  --device cuda --num-action-samples 32 \
  --eval-seed <declared-policy-seed> --env-seed <declared-reset-seed> \
  --video-point 0 --video-point 1 --video-point 2
```

Run that command through the root-owned `abc_bench.mulligan_program` canonical
supervisor, not as an unleased GPU job. Root owns source/runtime admission, the
checkpoint and start-manifest pins, normal release, native execution and result
publication. The recorder agent owns only this observation boundary and its CPU
checks. The launcher agent owns process supervision. See
[the architecture source](MULLIGAN_AUTHOR_VIDEO.mmd).

Installed CPU checks run the actual original evaluator, rollout and Normalizer
with explicit policy/world doubles and real FFmpeg MP4 encode/decode. They prove
observer behavior, not native success. Root separately measured original native
OSMesa render-on/off parity for 13 states and 12 zero-action steps in
`builds/mulligan-author-experiments-20261013/runtime/NATIVE_RENDER_PARITY.json`.
Learned-policy native evaluation and video acceptance remain root's next gates.
No native or GPU execution was performed by this recorder agent. A full
ASD-STE100 compliance check was not run.
