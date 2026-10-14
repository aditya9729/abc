# Short temporary paths for original FastTD3

The original one-call run `author-subset-20261014T024040-690158` failed with
`OSError: AF_UNIX path too long`. The supervisor set `TMPDIR` to the long run
output directory. Python's forkserver adds its own socket directory and name.
The failure occurred during the first SubprocVecEnv process start. It does
not establish native world allocation or a learning result. Keep that run
failed and unchanged.

The supervisor now creates one unique `/home/user/aditya/t/ft-*` directory.
It sets the child's `TMPDIR` to that short path. Original source, arguments,
128-world count, learner defaults, output CWD and lease rules are unchanged.

The worker receipt records `temporary_directory` and its cleanup owner.
The directory contains `owner.json` with the run output path and supervisor
PID. The supervisor retains it on success, failure and cancellation. Root
can remove that specific owned directory after canonical group cleanup.
The supervisor does not remove shared temporary files or live-child files.

The installed CPU regression first reproduces the long-path failure. It then
starts and joins a real public `multiprocessing` forkserver child through the
repaired supervisor, with the same deliberately long run output path. It
checks the short path, unique ownership and retained directory. No simulator,
model or GPU work is used. Native original FastTD3 retry remains root-owned.

The [diagram](FASTTD3_SHORT_TEMP_REPAIR.mmd) shows the boundary. The preparation
agent owns this patch and its CPU checks. Root owns normal release, original
retry and post-group cleanup. STE guidance was used; full compliance was not
checked.
