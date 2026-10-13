# Original Real-Time EXPO-FT entrypoints in ABC

The supervisor runs the author's Python entrypoints as child processes. It does
not implement a learner, sampling loop, replay schedule, or update rule. The
canonical runner owns the GPU0 lease and the outer timeout.

The declared four-file author patch adds dataset and public model-configuration
dispatch. The original loop bodies are unchanged. The ABC task changes the robot,
three camera inputs, action coordinates, demonstrations, normalization statistics,
and sparse success condition. This is an ABC task adaptation. It does not reproduce
the authors' DROID Dynamic Picking benchmark.

The first native check runs original `train_offline_rtc.py`, `RTCLearner`, B64,
seed42, maximum prefix delay10, and one update. It uses the real-time OpenPI
branch and fresh pi0.5 foundation weights. One update tests the training path;
it does not establish policy quality. The ordinary SFT125 actor is not used.

The process receipt does not accept a checkpoint. The author catches checkpoint
save exceptions, so a separate check must verify the saved actor, optimizer step,
schedule step and random state. Metrics stay in local W&B offline files. Nothing
is sent to a tracking service.

For environment jobs, the supervisor requests SIGINT so the task environment can
close resources at Python exit. It records escalation and attempts cleanup of
both children independently. Native cleanup and video completion require separate
evidence. The author client sanitizes nonfinite actions; environment checks alone
cannot prove that policy output was finite.

Root owns release, GPU use and result publication. The data agent owns the input
boundary. The environment agent owns the ABC client boundary. The review agent
checks method fidelity and checkpoint evidence. These roles exchange pinned files.
[Architecture](REALTIME_AUTHOR_PROGRAM.mmd). STE guidance used; full dictionary
and grammar compliance were not checked.

The native checks are now complete. See [measured results](REALTIME_NATIVE_RESULTS_20261013.md).
Install the ABC client task metadata dependency with `abc-minimal[realtime-expo]`.
The separate client prefix also contains the original real-time OpenPI client wheel.
It does not contain the server adapter or JAX. Runtime versions and wheel hashes
are recorded in the release evidence.
