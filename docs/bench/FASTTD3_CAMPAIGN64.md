# Explicit 64-world FastTD3 campaign

The RAM estimate refused the 128-world campaign before the native host started.
The separate `reward_v2_training64` profile keeps the existing conservative memory formula.
The measured worker maximum was 3,707,260 KiB.
The estimate is 304,969,408 KiB for 64 worlds and 601,550,208 KiB for 128 worlds.
Read current `MemAvailable` before admission. Do not replace the maximum with a mean.

The new gate collects 4,096 transitions from 64 calls.
It must produce 106 critic updates, 53 actor updates and 64 clean worker exits.
Training requires a new gate with the same world count, source and reward.
The historical 128-world gate does not admit this training profile.

The maximum stays 42,480 calls per world, with 2,718,720 total transitions.
The minimum stays 21,240 calls and six completed native episodes per world.
Task stopping still needs three complete 3,540-call windows without declared meaningful improvement.
This rule measures TRAIN-task futility. It does not establish validation convergence or overtraining.

The original learner settings remain fixed: B32768, per-world replay51200, two critic updates per call and gamma0.99.
The 64-world profile halves aggregate replay capacity from 6,553,600 to 3,276,800 transitions.
It doubles critic updates per collected transition from 2/128 to 2/64.
The original replay sample has 512 samples per world, compared with 256 for 128 worlds.
These are explicit recipe differences. This profile is an adapted ABC task diagnostic, not an exact paper reproduction.

The seed stride stays128. The first64 ranks keep their original per-rank scene sequences.
The reward, native judge, action map and physics failure checks stay unchanged.
All historical128 configurations and receipts remain separate.
Full-task success still has a long credit horizon compared with gamma0.99 at the ABC control rate.

The ABC agent owns native host and supervisor changes. The Fast peer owns the normal adapter.
The coordinator owns normal releases, the native gate, training and captured validation.
CPU tests use explicit simulator/process artifacts. They do not establish native task performance.

[Architecture source](FASTTD3_CAMPAIGN64_ARCHITECTURE.mmd) shows implemented interfaces and the required native workflow.
Native gate, training and frozen task results remain pending for this new profile.
This text uses STE guidance. It has not had a full ASD-STE100 compliance check.
