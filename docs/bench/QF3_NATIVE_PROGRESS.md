# QF3 ABC reproduction progress

The active scope is QF3 in native ABC, then Galaxea R1 Lite. R1 Pro is removed.
The earlier DiT output adapter did not match the paper's ABC-VLA experiment.
The current Sadhana implementation uses the native VLA action head with internal rank-four adapters.
It preserves the full 30-action flow context and executes the first 15 actions.

The [QF3 paper](https://arxiv.org/pdf/2610.08789v1) reports about 91% to 95% bottle-task success.
It reports 781 to 606 control steps in successful episodes, with three training seeds.
These remain reproduction targets. Our current evidence does not establish these results.

| Evidence | Actual result | Interpretation |
| --- | --- | --- |
| Installed native check, harness0.4.2 | 30 control ticks, three critic updates, one actor update | Model and simulator integration; independent weight/replay review passed |
| Frozen candidate baseline, 50 worlds | 197 ticks each; 9,850 world steps; 44 first placements; zero complete episodes | Deadline stop; success rate unavailable; all five bottles are required |
| Reviewed installed release, harness0.4.4 | 782 harness CPU tests; 66 ABC CPU tests; seven final dispatch tests; 39 installed files verified | Full cadence, timing and restore journal software; no native0.4.4 execution |

The released shared candidate is not confirmed as the paper's exact initial checkpoint.
Unpublished critic, seed, and replay choices are explicit in the source audit and configuration.
Partial episodes do not enter the success denominator. Successful episode length remains separate from failure-inclusive length.
No R1 Lite learned-policy performance has been measured.

Sadhana release commit: `5305645` on `staging_attempt1`.
Its source and independent reviews are under `/home/user/aditya/RL/sadhana-resfit/harness/`.
Use `docs/QF3_ABC_REPRODUCTION.md` for the architecture and team explanation.
Use `docs/QF3_TRAINING_RUNBOOK.md` for configuration, execution and checkpoint contracts.

The UI is read-only at `http://127.0.0.1:8766/`.
Its offline snapshot is `/home/user/aditya/RL/abc/outputs/bench/dashboard.html`.
The latest runtime row links to 16 source, review and progress artifacts.
Raw native receipts and checkpoints stay under `outputs/bench/qf3-vla-*`; they remain unchanged.

The original one-GPU, two-hour ledger has 78.6659 seconds left.
The next prepared plan has four short batch-size profiles and a full-horizon frozen baseline attempt.
Their maximum parent reservations total 3,180 seconds within a proposed additional one-hour allowance.
This plan requires explicit approval for extra compute. It contains no full three-seed training campaign.
The controller retains prior charges, checks GPU occupancy and holds one physical-GPU lease.
Leela jobs remain untouched.

This explanation uses ASD-STE100 guidance. Full vocabulary and grammar compliance was not checked.
