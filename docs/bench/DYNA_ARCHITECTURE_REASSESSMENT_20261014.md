# What Dyna's architecture requires

The published replay could not grasp a towel by construction. Our implementation disables contacts, has no dynamic joints or actuators, assigns recorded body poses, and calls `mj_forward` without physics integration. Towels are static visual geometry. This explains the failure independently of alignment. The video is a visual diagnostic; it does not complete the requested autonomous experiment. The accepted render receipts certify rendering and pose preservation only.

The [requested post](https://x.com/MarilynLiu97/status/2109808785615458626) discusses **Dyna-2.1 on Taku**. Its R1 Pro reference describes another team's work. Direct X access was blocked; the preserved official syndication response identifies the post and article, and a public article mirror supplied its text. Technical claims below use Dyna's primary publications.

**Dyna-2.1: the three layers.** The [official report, sections 2.2–3.2](https://www.dyna.co/dyna-2.1) describes:

| Layer | Published responsibility |
| --- | --- |
| Orchestrator | A post-trained vision-language model observes progress, keeps text memory, chooses steps, checks completion, and requests recovery. It runs more slowly than the policy; its exact rate is undisclosed. |
| Action policy | An improved Dyna-2 model generates task-space trajectories at approximately 5 Hz. Checks within a fold remain here. Controller metadata and queued future commands condition execution. |
| Whole-body controller | Simulation-trained RL maps URR targets to joint targets and wheel velocities at 100 Hz. URR specifies wrists, elbows, chest and footprint. Human and synthetic targets train tracking; dynamics are randomized and oscillation is discouraged. |

The action policy uses human video mixed with fleet data; the same motion targets support controller training. Isaac Lab is identified for **controller training**, not as a released laundry-cloth training benchmark. Taku has two seven-axis arms, a folding body and four steerable wheels. Figure 3.1 is a **teleoperation log**, not the autonomous workflow. [Source](https://www.dyna.co/dyna-2.1)

**Inside the action model.** [Dyna-2, section 2](https://www.dyna.co/dyna-2) describes separate video and action diffusion-transformer streams trained with flow matching. Video attention is causal; action attention is bidirectional. Proprioception enters the action stream. Language enters through video cross-attention. The shallower action stream connects to early video layers. In the reported scaling-study variant, future-video prediction is an auxiliary training objective: action inference uses observed context, not generated future video. This is not a live generative simulator. Dyna-2.1 calls its policy an improved Dyna-2; its exact changes are undisclosed.

Dyna-2's human-only scaling study differs from Dyna-2.1's mixed-data recipe. It derives wrist trajectories and a continuous grasp signal from hand-pose tracks and thumb–index aperture. Neither the million-hour corpus nor a complete action codec is released in the inspected materials. Do not infer that Taku uses that aperture directly as a physical jaw-width command. [Source](https://www.dyna.co/dyna-2)

**Sensing and grasping.** Dyna's [current system diagram](https://www.dyna.co/) names vision, tactile sensing and proprioception, but does not publish their tensor schemas, calibration, filtering or exact consumer connections. The [released kinematic tree](https://www.dyna.co/assets/we/dvt1_kin.json) includes wrist-camera frames and opposed prismatic fingers. The public viewer maps each normalized opening to opposite finger displacements of 0.0345 metres. These are geometry and visualization facts, not calibrated actuator limits, force control, tactile processing or evidence of a successful grasp.

**Learning from operation.** The [deployment report](https://www.dyna.co/research/scaling-customer-deployments) describes aligned recordings, automatic SOP/outcome/failure labels, human review, targeted data curation and new checkpoint evaluation. It distinguishes hardware wear from policy error. This does not specify online RL updates during the showcased laundry run. Whole-body tracking RL does not by itself learn cloth selection, grasp success or recovery.

**What our files actually represent.** The public recording is episode `62458843`, `WE 7DOF Dryer`, robot `dvt-robot-sh008`, dated 2026-09-21: 869 samples at 12 Hz. Its relation to the hour-long autonomous run remains unknown. Its normalized grip channels, odometry, joint states and URR targets are insufficient to reconstruct visual observations, cloth state, contacts, policy decisions or a 100 Hz controller. Resampling them cannot restore those missing signals. The estimated room and props added by us are not author assets.

| Exact reproduction gap | What is available / next boundary |
| --- | --- |
| Orchestrator | Description only; base model, checkpoint, prompts, memory format and post-training recipe unknown. |
| Action policy | Architecture description; no located author checkpoint, inference code, action codec, chunk horizon or deployment sensor schema. |
| Whole-body RL | No located checkpoint, algorithm, observation/history schema, reward weights, training budget or actuator model. PPO is not established. |
| Robot physics and grippers | Visual model and joint tree; masses, inertias, limits, servo gains/delays, pad friction and sensing need author release or labelled estimates. |
| Laundry | Reference media and partial dryer metadata; no located original deformable assets, room physics, reset bank, cloth recipe or autonomous episode data. |

Release search on 2026-10-14 inspected official report links, GitHub repository metadata searches, and the public profile named `dynarobotics` on Hugging Face, which lists no public models or datasets. That profile's author affiliation is not independently established. This is bounded negative evidence, not proof that no release exists. Downloaded evidence and hashes are in [source evidence](DYNA_ARCHITECTURE_REASSESSMENT_20261014.json); the exact rate labels are retained in [the original figure](/home/user/aditya/RL/builds/post-scene-reproduction-20261014/dyna-architecture-study-01/sources/figure-2.4-original.svg).

**R1 Lite implication.** The [published local transfer assessment](R1LITE_AUTHOR_RECIPE_TRANSFER_20261014.md) establishes 14 absolute arm/jaw targets at 15 Hz, 40 physics steps per control at 600 Hz, position servos, and a fixed base/torso for the first task. The 600 Hz value is the physics integration rate, not a verified separate servo-control rate. It is not a URR whole-body interface. Preserve that existing boundary. A future task-space adapter must solve reachability, frames, timing and jaw commands; mobile whole-body control also needs new base/torso interfaces and validation. It cannot be added by renaming the current 14-action vector or changing its rate to 100 Hz.

The next useful deliverable is the [physical grasp experiment](DYNA_PHYSICAL_GRASP_PLAN_20261014.md), explicitly a local experiment. The requested autonomous full-room experiment remains outstanding beyond that preflight. [Architecture source](DYNA_ARCHITECTURE_REASSESSMENT_20261014.mmd) separates Dyna's described system from components we have and components we must build. The scene agent owns this study; root owns implementation admission and acceptance; Leela owns published simulation boundaries. No native run, install, training or viewer change occurred. STE guidance used; full compliance not checked.
