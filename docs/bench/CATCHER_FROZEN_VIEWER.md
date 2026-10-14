# Optional frozen Catcher development panel

This viewer addition has no native results. Root owns result acceptance and publication.
The existing BC, training, Mulligan, and earlier ABC panels remain available.
No `catcher.frozen_comparison` value hides the new panel.
The panel requires all eight accepted cases before it shows scores or videos.

Root may add this object to the existing results snapshot:

```text
catcher.frozen_comparison:
  schema: catcher.frozen-comparison/v1
  status: accepted_matched_development
  rows: four rows ordered by seed 100001, 100002, 100003, 100004

each row:
  seed: integer
  rtc_bc24: accepted case
  saved_r2: accepted case

each accepted case:
  status: accepted_frozen_case
  solved: boolean native outcome
  controls: integer, 1 through 256
  frames: controls + 1, including the reset frame
  complete: true
  sha256: full lowercase SHA256 of the published MP4
  video: first 16 hash characters + .mp4
```

These labels describe root acceptance. The browser does not authenticate producer receipts or video bytes.
Root must validate full restore, zero updates, native episodes, matched starts, videos, and clean process exits.
Keep accepted checkpoint files unchanged. Reject partial, failed, or unrun cases before publication.

The baseline uses genuine BC24 through the original critic-free RTC learner.
The learned arm uses the short R2 whole policy, including candidate selection and editing.
Both use delay 4, C4/H8, and action noise 0.1 on four development seeds.
The task/backend hooks and Catcher-only post-fill scheduling recheck remain declared adaptations.
This comparison does not reproduce paper performance or isolate one learned component.
Each video is a complete native episode at 30 simulation frames per second.
Playback does not prove wall-clock real-time execution.

CPU checks use explicit DOM and media fixtures. They are not native results or browser playback proof.
Run the actual template checks without installing dependencies:

```sh
/home/user/aditya/runtimes/node-v22.16.0-linux-x64/bin/node --test tests/bench/test_catcher_frozen_viewer.cjs
```

Root must check actual published videos and desktop/mobile playback after native acceptance.
The static exporter already preserves result JSON and existing media. It is unchanged.
See [the workflow](CATCHER_FROZEN_VIEWER.mmd). Root owns native execution and publication.
This agent owns the optional rendering change. Author code owns policy sampling and evaluation.
The text uses STE guidance. Full ASD-STE100 compliance was not checked.
