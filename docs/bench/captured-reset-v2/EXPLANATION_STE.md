# Explanation with STE guidance

The root agent owns the simulator release and GPU jobs.
The EXPO setup agent owns this host and client change.
The independent reviewer checks the release evidence.

The cohort identifies each validation episode.
The host checks the episode and completed collector receipts.
The host loads the saved model.
It restores the complete saved integration state.
This state includes the carried arm state.
The host checks the model, physics state and task judge.
It renders three actual camera images.
It compares their bytes with the recorded initial images.
It checks the physics state again.
A failed check disables actions.

The client checks the same episode and reset proof.
It receives actual camera images before the first policy action.
The host submits original float32 commands without host clipping.
Native actuator limits still apply.
The historical profile retains its recorded clipping behavior.

CPU tests check restoration and rejected inputs.
They use explicit task and camera doubles.
They do not prove actual camera behavior or task success.
The root agent must run the live camera acceptance step.
It must then run the selected frozen policy.

This text uses ASD-STE100 guidance.
It has not received a full dictionary and grammar compliance check.
