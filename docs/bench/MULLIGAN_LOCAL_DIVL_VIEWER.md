# Optional local DIVL viewer contract

This rendering change contains no native results. Root owns acceptance and publication.
The existing local IDQL, released-policy, Catcher, and earlier ABC sections remain separate.
An absent `mulligan.local_divl` key hides the new section.

```text
mulligan.local_divl:
  status: accepted_local_checkpoint
  stage: divl_head
  num_action_samples: 32
  scores:
    - n: 1, episodes: 50, successes: actual integer 0..50
    - n: 32, episodes: 50, successes: actual integer 0..50
  clips: exactly three objects, ordered by fixed point 0, 1, 2

each clip:
  point: 0, 1, or 2
  complete: true
  solved: actual boolean outcome
  controls: actual integer 1..400
  frames: controls + 1, including reset
  sha256: full lowercase MP4 SHA256
  video: first 16 hash characters + .mp4
```

Root must validate genuine producers, matched starts, full episodes, video decoding, and clean exits before publishing this key.
The browser validates presentation metadata. It does not authenticate producer receipts or video bytes.
The summary derives both scores and declares the combined candidate-count/ranking change.
The new section identifies DIVL as a later frozen-actor stage. It does not isolate critic causality.
Original 400-start monitoring scores remain separate from the 50 viewer starts.
Complete video playback uses 20 simulation frames per second, not wall-clock control time.

Four CPU DOM-fixture checks run with the existing Node executable:

```sh
/home/user/aditya/runtimes/node-v22.16.0-linux-x64/bin/node --test tests/bench/test_mulligan_local_divl_viewer.cjs
```

The fixtures prove rendering and retention behavior only. Root must inspect actual published media in the browser.
The exporter, learner, actor, controller, and native runtime are unchanged.
Root owns execution and release. This agent owns the optional renderer extension.
See [the workflow](MULLIGAN_LOCAL_DIVL_VIEWER.mmd). STE guidance used; full compliance was not checked.
