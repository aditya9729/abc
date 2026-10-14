# Original-program client shutdown

The supervisor previously sent SIGINT as soon as the author process exited.
This could interrupt a client that was already closing its connection and native environment.
It also left status `completed` when the client then exited nonzero.

The observed Catcher run `author-subset-20261014T053638-6a599b` saved checkpoint4096 and exited author0/client-2.
That run and its interrupted-client recovery evidence remain unchanged.
A shutdown race is plausible; the exact signal-delivery site was not recorded.

The supervisor now gives the client five seconds to exit naturally.
An exit during this grace uses the existing shutdown label `already_stopped`.
If needed, the original SIGINT20s, SIGTERM10s, and SIGKILL10s waits follow.
The total client cleanup bound is45s. The existing parent cleanup margin remains unchanged.
An owned child that disappears before a signal is reaped without signalling another process.

Any nonzero client exit now prevents completed status.
The receipt retains the actual author and client exit codes.
An author0 checkpoint remains an artifact for separate inspection, with `checkpoint_accepted=false`.
The supervisor does not convert that artifact into a successful training job.
It does not change the author learner, client, replay, control loop, or task settings.

`tests/bench/test_author_client_shutdown.py` uses real bounded CPU subprocesses and explicit process stand-ins.
It checks natural exit without SIGINT, original hanging-client escalation, and nonzero exit handling.
It also checks supervisor-generated SIGINT/client-2 and deterministic post-reap races.
The race fixture reaps its real owned child, then raises an explicit ProcessLookupError.
No test claims a native robot, simulation, GPU, checkpoint, or learning result.

The isolated wheel test uses a normal installation in a separate CPU prefix.
The Git-only wheel has27 tracked assets. It is not a complete native ABC release.
Root owns the full-asset release and the next actual process check.
Five seconds is a chosen finite grace, not a measured upper bound for all client finalization.

[Architecture](REALTIME_CLIENT_SHUTDOWN_REPAIR.mmd).
Root owns release and actual process acceptance. The author owns learning and control.
This agent owns the supervisor repair and CPU checks.
STE guidance used; full ASD-STE100 compliance was not checked.
