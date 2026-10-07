# QF3 native ABC progress

The frozen public ABC-VLA candidate completed 50 bottle episodes. Ten succeeded, or 20%.
Successful episodes averaged 935.9 ticks. Forty episodes timed out at 1,000 ticks.
The native current-placement judge reported seven successes. The paper's history rule counts first-ever placement of all five bottles.
Independent review accepts the saved result and unchanged learner state.
This does not reproduce the paper's starting results: Figure 6 pools about 91.33%; Appendix A.7 reports a separate matched 86% baseline.

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
Timing profiles used installed harness 0.4.7 with 824 CPU tests. The later recovery experiment used 0.4.8 with 901 builder CPU tests and 254 independent focused checks. ABC passed 143 CPU tests.
All 39 installed Python members were independently verified.
A full16-world cohort completed16,000steps in225.147parent seconds. A short80-world check completed1,200steps in59.316seconds.
The80-world check remains partial at15ticks per world. It does not prove combined80/16/50allocation or complete warmup.
Installed 0.4.8 passed actual completed-boundary recovery. A new process restored the saved model, optimizers, learner RNG and all 67 input replay records before updates. The final chain completed 2000 warmup, 2000 training and 2000 validation ticks, four critic updates, two actor updates and 268 replay records.
Two complete fresh one-world validation events have distinct public reset metadata and preserve training state. Neither succeeded. They evaluate different actor checkpoints and do not form one learned-policy score.
The reference also restarted. Repeated trajectories and final weights differ. Partial native worlds are discarded and reset; exact physical replay and object-geometry freshness remain unverified.
Current installed 0.4.9 repairs standalone evaluation reporting and phase accounting. In 0.4.8, inherited validation summaries could appear without a new evaluation, and checkpoint time overlapped evaluation time. The repair passed 908 full CPU tests in both builder and independent runs. All 39 installed files match the reviewed wheel. No native GPU experiment has run under 0.4.9. Parent wall charge is authoritative.
Full-ring replay/checkpoint storage and combined environment memory need measurement before a long campaign.

Source and review documents are under `/home/user/aditya/RL/sadhana-resfit/harness/docs/`.
Use `QF3_ABC_REPRODUCTION.md` for the architecture and team explanation.
Use `QF3_CORRECTED_BASELINE_REVIEW.md` for current metrics and their limits.
Use `QF3_FULL_OPTIMIZER_PROFILE_REVIEW.md` and `QF3_CAPACITY_PROFILE_REVIEW.md` for measured costs.
Use `QF3_MEASURED_TRAINING_PLAN.md` for conditional scenarios, not guaranteed duration or new compute authorization.
Use `QF3_NATIVE_RECOVERY_REVIEW.md` for actual recovery proof and numerical divergence.
Use `QF3_BASELINE_GAP.md` for primary-source evidence about the unknown author checkpoint, scene, runtime and prefix schedule.
Raw receipts remain under `outputs/bench/qf3-vla-*`. Historical wrappers remain unchanged.
The dashboard publishes a separate reviewed baseline row with 10/50 successes.
There are 49 canonical rows. The reviewed 0.4.8 recovery diagnostic has unavailable performance values; its 38 live/offline artifact links passed independent review. Two additive corrected copies preserve the original snapshot and fix stale architecture and resource-account wording.
The new 0.4.9 software row has 21 artifact links and no native performance values. It records the installed reporting repair and independently reviewed next-run proposal. All 21 live/local/offline links passed independent publication review. Prior receipts and snapshots remain unchanged. See `QF3_REPORTING_PUBLICATION_REVIEW.md` in the Sadhana docs for scope and pins.
The UI is read-only at `http://127.0.0.1:8766/`.
Its offline snapshot is `/home/user/aditya/RL/abc/outputs/bench/dashboard.html`.
R1 Lite remains the later policy target. R1 Pro is removed from active work.

The user approved one additional GPU hour. The cumulative ledger limit is 10,800 seconds.
At this publication, 10,538.595722 seconds are charged and 261.404278 seconds remain. All four recovery jobs are reaped.
No GPU job is active. Full three-seed training is not included in this extension.
The controller keeps prior charges, checks occupancy and holds one physical-GPU lease.
Leela's jobs remain untouched.
The new proposed calibration uses full 80-world/four-batch warmup, one 16-world rollout, 1600 critic and 200 actor updates, and 50 fresh validation layouts. It requests a separate additional 7200-second allowance, which has not been approved. The matched fixed-development50 evaluator is conditional on at least 600 actual settled seconds remaining and a real accepted checkpoint. See `QF3_NEXT_RUN.md` and `QF3_NEXT_CAPACITY_REVIEW.md` in the Sadhana docs. This calibration is separate from full 400k/three-seed reproduction.
This explanation uses ASD-STE100 guidance. Full vocabulary and grammar compliance was not checked.
