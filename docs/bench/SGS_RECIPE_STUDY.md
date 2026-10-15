# SGS recipe study: implications for ABC and R1 Lite

Study date: 2026-10-12. Sources: [paper v1](https://arxiv.org/html/2610.12465v1)
and [author project](https://sgs-rl.github.io/).
This is a design study. No SGS implementation or training run is claimed.

## What the recipe changes

The project describes an outer loop around PPO. It selects episode reset
configurations using recent success. Intermediate success receives more weight;
easy and hard configurations retain coverage. The reset bank includes reaching,
stable-grasp and near-goal states. Code is marked as coming soon.

Paper sections 3.2 and B.2 specify a fixed configuration bank and 100-outcome
histories. Manipulation uses N=32768, target=0.5, concentration=1, epsilon=0.0001
and temperature=2. The appendix additionally floors weights and clamps temperature
to at least one. Empty-history numerical initialization is unspecified.
Locomotion epsilon differs between section 3.2 and the nominal marking in table 3.

In the rod-in-hole reset ablation, removing near-goal states gave zero success
on the remaining reset families. SGS selects available states; it does not create
them. The transfer recipe uses a point-cloud teacher and an RGB DAgger student.
Data from held-out student environments is excluded from gradient updates.
These details are separate from sampler weighting. See appendices B.3 and C.3.

The complete recipe also needs attention to control and contact physics.
UR5e uses Cartesian operational-space control; Franka uses joint-position control.
The main manipulation runs use a gravity curriculum. The no-curriculum ablation
uses one training seed, so it does not establish equivalence across tasks.
Transfer experiments reduce gripper-pad friction to avoid unrealistic grasps.
They also match object symmetries in resets and success criteria.
These are additional recipe choices, not effects proved for the sampler alone.
See appendices A.2, B.4 and C.2–C.3.

## Our interpretation

Our FastTD3 pilot collected no positive bottle-placement reward. If every
configuration has a zero estimated success rate, SGS gives them equal weights.
The sampler cannot distinguish those configurations or create a new reset state.
This follows from the equations. It is not a measured SGS result in ABC.

The immediate design question is whether our simulator can expose useful,
physically valid intermediate states. Proposed ABC states are a bottle already
held, a bottle close to placement, and a full pickup layout. These are proposed
task analogies. They are not the author's assembly reset implementation.

Each proposed state must include consistent robot and object position, velocity,
contact, gripper configuration, geometry and task history. Validate those states
with the original physics and success judge. An initial completed task must not
count as newly learned behavior. Changing resets also changes the training
distribution. Keep that change explicit.

For a future study, compare uniform and adaptive sampling from the same TRAIN
bank. Hold the learner, parallel world count, update schedule and compute budget
constant. Evaluate on fresh full-task scenes. Keep evaluation outcomes out of the
sampler and early-stopping decisions. Log success and sample counts per reset
family so a near-placement skill cannot hide failed full-task pickup.

For R1 Lite, Leela could publish validated reset states as versioned scene
artifacts. Sadhana could own their training allocation. Atlas and Engram could
provide reviewed source trajectories when their interfaces are specified.
These are proposed handoffs. No private project imports or verified integrations
are implied.

FastTD3 plus SGS would be a new off-policy experiment. Replay retains older
mixtures of configurations, so its data distribution also needs inspection.
The paper's PPO results do not prove that combination. EXPO-FT or a VLA residual
would require their own comparisons. The one-world FastTD3 reuse mismatch remains
even if reset allocation improves.

Root studies the design and records these limits. The FastTD3 peer checks the
sampler equations and source availability. No new GPU experiment was launched
for this study. [Diagram source](SGS_RECIPE_STUDY.mmd) distinguishes the paper
workflow from a proposed ABC connection. Text uses STE guidance. No full
ASD-STE100 compliance check was done.

## R1 recovery and configuration history — 15 October 2026

The user highlighted reuse of past success for each configuration. SGS stores
recent success and failure outcomes for each fixed reset configuration. It uses
those outcomes to allocate new episodes. It does not retrieve a successful
trajectory or supply observation history to the acting policy. The paper's
configuration includes the initial state, goal and environment. Event-based
grouping would be an additional design choice. Sections 3.1–3.2 and B.2 support
these distinctions; the author project still marks code as coming soon today.

For the R1 recovery study, retain configuration IDs, source lineage and versioned
outcomes with each episode. Keep three interventions separate: adaptive reset
selection, replay sampling of successful experience, and policy history. Start
with the fixed-input EXPO identity and selection audit. Then validate independent
corrections and reset coverage before comparing uniform with guided collection.
Use the same TRAIN bank and learning budget; keep evaluation outcomes outside
the sampler. Connecting SGS allocation to EXPO remains an explicit off-policy
extension, with both collected and sampled replay mixtures measured. No sampler,
learner, reward, memory architecture or runtime was changed by this update.
