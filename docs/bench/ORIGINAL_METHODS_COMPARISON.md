# What the current results can establish

FastTD3 needs a task reward from the simulator. It learns two action-value critics and a policy. It does not need a supplied pretrained value model. The author learner consumes environment rewards in `fast_td3/train.py`; it preserves terminal observations and bootstraps time-limit truncations.

| Item | FastTD3 in this harness | EXPO-FT in this harness |
| --- | --- | --- |
| Initial policy | Random state policy | Pretrained pi0.5, then task SFT |
| Observation | Privileged robot and object state, 216 numbers | Three camera views and robot state14 |
| Demonstrations | None | Three complete TRAIN episodes |
| Training signal | Explicit adapted environment reward | Binary full success; complete demos seed its replay |
| Learner | Original FastTD3 | Original EXPO with original OpenPI actor |
| Simulator result | Native YAM five-bottle judge | Same native YAM five-bottle judge |
| Frozen evaluation | Captured VAL state, no learning | Same captured VAL state, no learning |

This is a comparison of complete training strategies. It cannot isolate which RL algorithm is better. A matched algorithm experiment would also have to match the starting policy, observations, data, action contract and training budget. This work does not make that claim.

The completed FastTD3 run used 128 worlds and 4000 vector calls: 512000 transitions, 7978 critic updates and 3989 actor updates. Each world completed only one full TRAIN episode. All 128 completed episodes failed the five-bottle task. Frozen VAL9100 also failed:0/1 episode,0/5 bottles. There was one simulator acceleration warning. These observations do not establish convergence or a failure of the original paper.

The historical task reward paid the current bottle fraction every control. With discount 0.99, one bottle held for100 controls pays 12.679 discounted units; immediate full success with termination pays1. This is a reward-design problem. The result does not prove that the policy deliberately exploited it.

The corrected experiment keeps the original learner and judge. It uses binary full success plus the versioned shaping term `0.99 * Phi(next) - Phi(current)`. True success terminals use zero next potential. Timeouts retain the actual final potential and original bootstrap semantics. Phi is bounded by 0.25 and uses current placement plus reach, lift and transport geometry. These are disclosed initial heuristics. They are not author-supplied ABC rewards and have not been selected with VAL outcomes.

This correction removes repeated partial-placement payments. It does not guarantee useful exploration, stable physics or task success. The short128-world gate checks the actual geometry, reward and learner interface. The long bounded campaign then measures TRAIN task progress and can stop for lack of progress after a minimum of six complete episodes per world. The final captured VAL episodes remain separate from this stopping rule.

The original discount 0.99 also needs attention. At 0.034seconds per control its e-fold horizon is 3.383seconds. Successful TRAIN demonstrations take 35.258–47.940seconds. Their final unit reward contributes only about 3.01e-5 to 7.08e-7 at the first transition. This is a discount-to-task-duration mismatch, not proof that the learner is defective. With no full-success samples, bounded potential shaping cannot guarantee that the agent discovers grasping or learns a useful task value. The gamma and reward defaults remain explicit; no task-tuned setting has been silently introduced. Exact calculations and source pins are in [the calculation receipt](/home/user/aditya/RL/builds/abc-fasttd3-reward-design-v1/HORIZON_EXPLORATION_CHECK.json).

The author default has 150000 vector calls. The new ABC maximum is 42480, with TRAIN task stopping no earlier than 21240. This is a limited task adaptation. It is not a full reproduction of the paper. Both reward and budget differ from the historical 4000-call experiment, so any change is not a reward-only causal result.

The current pi0.5 supervised run has a 100-update bound and validates every complete held-out window every 25 updates. Its flow loss measures agreement with demonstrations. It is not bottle-placement success. EXPO's completed full-batch capacity call ran 20 critic/encoder updates and one actor/editor/temperature update. The state was discarded, zero online controls were issued and there is no online EXPO task score yet.

The original EXPO online schedule has ten completed warmup episodes. The planned eleven-episode run reaches its first three-call update trigger. This is an executed integration and first-update experiment when it completes, not evidence of convergence. It must fit with the live cameras and the original one-ahead batch on the single allowed GPU before the long run starts.

Both adapters use the same native absolute-position servos. Each control holds one target for 17 physics steps of 0.002 seconds. Fast maps normalized outputs to physical joint ranges and gripper fractions in [0,1]. EXPO forwards raw physical targets. A zero normalized Fast action selects the range midpoint; it does not hold the current position. EXPO replans after eight controls. These action-boundary and planning differences are part of the comparison.

The EXPO reward adapter already uses binary full success for both demonstration replay and live training. It preserves the author mask at any done, including timeout. Its original discount is also 0.99 per control transition. The author DROID pick task runs at 10 Hz with an 80-control limit; ABC runs at about 29.41 Hz with a 3540-control limit. Equal numeric discounts therefore do not provide equal horizons in seconds. The task and timing changes remain explicit adaptations.

Author reference: [FastTD3 repository](https://github.com/younggyoseo/FastTD3). The author documentation also describes task-specific settings and suggests investigating update count, multi-step returns and clipped double-Q if training stalls. Those settings have not been silently changed here.

This text follows ASD-STE100 guidance. A formal compliance check has not been done. Current evidence remains in canonical run receipts; this document is an explanation, not benchmark acceptance.
