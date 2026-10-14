# Proposed first physical cloth experiment

Goal: Taku's physical fingers acquire a free towel, lift it, hold it, release it, and recover after the towel moves. This tests the missing interaction, before full laundry orchestration. It is a new local experiment, not a Dyna policy reproduction.

Use a separate normally installed runtime and fresh results directory. Start with CPU MuJoCo 3.3.7: its [2D flex objects](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#deformable-objects) support deformation and frictional contacts; the official [pancake example](https://github.com/google-deepmind/mujoco/blob/3.3.7/model/flex/pancake.xml) supplies a bending/contact starting point. These examples do not establish towel realism or grasp reliability. Validate those locally.

Retain published Taku geometry and joint frames. First hold the base and torso fixed, and control one articulated arm and its fingers with bounded actuators. Record all estimated dynamics. Use simplified collision pads aligned to the published fingertips, a table, and one 0.30 by 0.30 metre free towel with an initial 21 by 21 mesh. A 0.09 kg mass, 1 mm contact radius and 1 ms physics step are initial **test choices**, not Dyna parameters. Fit or vary bending, stretch, friction and pad compliance after passive tests. Preserve the existing full-room visual experiment unchanged.

The first controller is an explicitly labelled diagnostic: an observation-driven state machine proposes wrist poses and jaw targets; feedback tracking drives physical joints. Start with privileged cloth state only to isolate contact mechanics, then replace it with rendered-camera corner/edge estimates. Keep oracle and camera results separate. A later Dyna-inspired learner must replace the diagnostic skill and controller with separately trained modules; no substitute is named Dyna.

| Stage | Required measurement and proposed acceptance |
| --- | --- |
| Passive cloth | Drop, drape and slide; no unexplained energy growth, solver warnings or penetration beyond contact thickness. Log cloth strain, energy and contact forces. Compare 1 ms versus 0.5 ms and 21×21 versus 31×31 before trusting a grasp. |
| Acquire | Observe an edge, approach, close, and confirm opposing finger contacts plus jaw/cloth consistency for 0.1 s. A closed empty jaw is failure. The actor cannot read the evaluator's success flag. |
| Lift and hold | Grasped patch rises at least 0.15 m, cloth centre rises at least 0.05 m, and the cloth stays supported by finger contact for 2 s. Report displacement/slip, contact force, strain and failed attempts. |
| Release | Open jaws above the table; cloth separates and settles under gravity. No cloth point may be welded, pinned, parented, teleported or continuously position-written to the gripper. |
| Recovery | Before closure, apply a bounded lateral physical push that moves the towel about 5 cm. Observe the new state, revise the target, retry at most twice, and record detection time, retries and final lift/hold/release outcome. Use a force pulse; do not relocate vertices. |

Run one bounded contact smoke first: two CPU cores, CUDA hidden, no more than 600 seconds wall time, after root source review. No native command is admitted by this study. If it passes, freeze material/controller settings and run 10 development resets, then 20 held-out resets paired between open-loop targeting and feedback recovery. Vary towel translation, yaw and small folds with saved seeds. Report all denominators, interventions and failures; a demonstration video alone is insufficient.

Record timestamped observations, proposed/applied actions, joint state, jaw width, cloth vertices for evaluation, contacts, forces, solver warnings and state transitions. Save one complete nominal run and one complete displaced-towel recovery run, including failed grasps. Privileged evaluator state must remain outside a camera-based actor.

Newton is a fallback if the MuJoCo contact test fails a documented stability or performance requirement. Its [original cloth/Franka example, pinned commit b7a4e8c](https://github.com/newton-physics/newton/blob/b7a4e8c99abaf036b91a57ee924adff80ccf77b6/newton/examples/cloth/example_cloth_franka.py) uses VBD cloth contacts, but prescribed robot targets and disabled robot gravity limit its use as a dynamics reference. It is not a learned cloth policy and supplies no Taku embodiment. Do not change engines merely to obtain a plausible video. Any install, GPU use or new solver execution needs a separate root admission.
