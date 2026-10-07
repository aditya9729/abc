# QF3 native ABC progress

The frozen public ABC-VLA candidate completed 50 bottle episodes. Ten succeeded, or 20%.
Successful episodes averaged 935.9 ticks. Forty episodes timed out at 1,000 ticks.
The native current-placement judge reported seven successes. The paper's history rule counts first-ever placement of all five bottles.
Independent review accepts the saved result and unchanged learner state.
This does not reproduce the paper's roughly 91% starting baseline.

The actual small learner check ran 30 ticks, three critic updates and one actor update.
The full optimizer timing check ran 1,600 critic updates at batch64 and 200 actor updates at batch256, plus a restore probe.
Optimizer phases took 257.552 seconds; parent wall time was 303.189 seconds.
It used two short zero-reward records and completed no episode. Full learning and the paper's three-seed result remain pending.
The public candidate is not confirmed as the author initializer.
Stock public inference and the bridge produce exactly equal full30 outputs for a retained raw observation and explicit noise.
Capture is now 168×224, followed by model padding/resizing to224. Earlier baselines captured square224×224 images.
Both camera runs reached10/50, but only three layout seeds succeeded in both. Their successful-only means do not show a speed gain.
The camera correction did not close the author-baseline gap. Universal parity remains unverified.

The collector has four native batch-size profiles and one matched optimization profile.
The 50-world matched run reduced bookkeeping from 35.845 to 0.866 seconds.
Submitted controls and task events matched exactly. Final states and image features differed.
This does not establish bitwise simulation parity or a full training speed claim.
Installed harness0.4.7 passed824 CPU tests. ABC passed143 CPU tests.
All 39 installed Python members were independently verified.
A full16-world cohort completed16,000steps in225.147parent seconds. A short80-world check completed1,200steps in59.316seconds.
The80-world check remains partial at15ticks per world. It does not prove combined80/16/50allocation or complete warmup.
Completed warmup checkpoint recovery passed852builder CPU tests and205independent focused checks; source is integrated but not installed.
Fresh validation source is under final independent CPU review. Native new-process recovery remains unverified.
Full-ring replay/checkpoint storage and combined environment memory need measurement before a long campaign.

Source and review documents are under `/home/user/aditya/RL/sadhana-resfit/harness/docs/`.
Use `QF3_ABC_REPRODUCTION.md` for the architecture and team explanation.
Use `QF3_CORRECTED_BASELINE_REVIEW.md` for current metrics and their limits.
Use `QF3_FULL_OPTIMIZER_PROFILE_REVIEW.md` and `QF3_CAPACITY_PROFILE_REVIEW.md` for measured costs.
Use `QF3_MEASURED_TRAINING_PLAN.md` for conditional scenarios, not guaranteed duration or new compute authorization.
Raw receipts remain under `outputs/bench/qf3-vla-*`. Historical wrappers remain unchanged.
The dashboard publishes a separate reviewed baseline row with 10/50 successes.
There are43canonical rows. All27new live/offline artifact links and the camera preview pixels are independently verified.
The UI is read-only at `http://127.0.0.1:8766/`.
Its offline snapshot is `/home/user/aditya/RL/abc/outputs/bench/dashboard.html`.
R1 Lite remains the later policy target. R1 Pro is removed from active work.

The user approved one additional GPU hour. The cumulative ledger limit is 10,800 seconds.
At this publication, 9,003.465442 seconds are charged and 1,796.534558 seconds remain.
No GPU job is active. Full three-seed training is not included in this extension.
The controller keeps prior charges, checks occupancy and holds one physical-GPU lease.
Leela's jobs remain untouched.
This explanation uses ASD-STE100 guidance. Full vocabulary and grammar compliance was not checked.
