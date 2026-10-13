# Original Real-Time EXPO-FT experiments: execution availability

The user requested a visible reproduction on the authors' experiments.
No local Real-Time EXPO-FT reproduction was executed in this check. The
available public release does not yet provide a verified executable path
for its Kinetix RL experiment. The released robot workflow needs physical
DROID equipment and task assets. This is an availability finding for the
checked sources, not a claim about private or unlinked releases.

## Visible author evidence

[Watch the full Dynamic Picking evaluation](https://pd-perry.github.io/real-time-expo-ft/static/videos/eval_videos/dynamic_pick/ours.mp4).
The recording is 319.894 seconds, at 1280 by 720 pixels. Root checked actual
browser seeking and playback after 200 seconds. A frame at 30 seconds shows
a robot and block on a rotating platform. This is the authors' recording.
It is not an experiment performed here.

[Watch their training recording](https://pd-perry.github.io/real-time-expo-ft/static/videos/training/dynamic_pick.mp4).
The [project page](https://pd-perry.github.io/real-time-expo-ft/#training)
labels training playback as 5 times capture speed. Its video stream lasts
144.467 seconds. A longer data track makes the container duration 722.342
seconds; that duration must not be presented as moving-video duration.
Neither downloaded file was edited.

Both files, checksums, media metadata, and playback evidence are under
`/home/user/aditya/RL/builds/realtime-expo-author-experiments-20261013/reference-videos/`.
Their total size is 27,232,680 bytes. Existing local rollout records remain
unchanged. No author video was inserted into our measured benchmark results.

## What can be reproduced from the checked release

| Experiment | Available | Required before local execution |
| --- | --- | --- |
| Original DROID Real-Time EXPO-FT | RTC-SFT, online learner, client, and evaluation programs | Connected and calibrated DROID robot, task equipment, demonstrations, statistics, and task-trained policy |
| Real-Time EXPO-FT on Kinetix | Paper settings; referenced simulator and pretrained baseline policies | Original Kinetix RL driver/configuration was not found in the checked release |
| Original RTC Kinetix baseline | Source and public per-level imitation-policy assets | Separate runtime and selected policy download; this alone cannot reproduce EXPO RL |

The official EXPO-FT repository remains at
`21fe3d3b7d913c80836496817965932f49e0aedf`. Both public branches, tags, releases,
and complete source trees were checked. The separately named
`pd-perry/real-time-expo-ft` repository contains the website and videos.
The author EXPO and FASTER repositories and linked coauthor FASTER releases
also did not provide the missing Kinetix experiment in this bounded search.

The original robot scripts use local data and checkpoint paths. Their
conversion script disables Hub upload. No verified public author task-trained
RTC or final online Dynamic Picking checkpoint was found. Missing and private
Hub repositories cannot be distinguished from the observed access responses.
The released task configuration also describes randomized object drops;
the website footage shows a rotating platform. Preserve this distinction.

The referenced [original RTC repository](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
was cloned unchanged at `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`.
It occupies about 1.57 MB. Public metadata confirms approximately 12.29 MB
per-level BC policies. No policy or million-transition dataset was downloaded.
The submodule and runtime remain uninstalled.

Source inspection also found a stale README evaluation flag. The current
entrypoint takes `--run-path`, not `--config.run-path`. Its CLI runs 96
settings per level and discards the rendered video returned by its evaluation
function. No command from this source review was claimed to have executed.

## Exact next requirement

For the user's requested original implementation, obtain the authors' Kinetix
Real-Time EXPO-FT driver and resolved task configuration. The published RTC
base assets can supply the initialization once their compatibility is verified.
A final RL checkpoint would permit evaluation before reproducing training.

Reconstructing the missing simulation integration from the paper is a separate
implementation. It must disclose all code and recipe differences. It cannot be
labelled an execution of the authors' missing Kinetix implementation. No such
substitution was made, and no new GPU or simulation job was launched.

If the user selects a paper-based reproduction, the proposed first experiment
is `catcher_v3`, one of the ten published Kinetix tasks. Reuse the original
level, simulator, pretrained state-flow policy, and observation preprocessing.
Keep the paper's action noise 0.1, chunk length 8, delay 4, replan length 4,
and ten inference denoising steps. Implement and review the missing state-based
RL integration explicitly; do not call it unchanged author code.

First check the published baseline and prefix alignment on that task. Then
train one seed with the paper's 100,000-transition budget. Record complete
baseline and post-training episodes under the same evaluation conditions.
Keep training statistics, checkpoint selection, and final evaluation separate.
The paper's four-seed result requires a later full comparison; one seed cannot
establish it. This route awaits clarification of the original-code requirement.

## Evidence and team

Detailed reports are under
`/home/user/aditya/RL/builds/realtime-expo-author-experiments-20261013/`:
`source-review/REVIEW.md`, `assets-review/README.md`,
`kinetix-readiness/READINESS.md`, and `HF_ACCESS_CHECK.json`.
The Kinetix readiness note corrects the README command quoted in the broader
source review. Root owns execution decisions and visual provenance. Separate
reviewers checked source availability, task assets, and the original RTC API.

[Availability diagram](REALTIME_AUTHOR_EXPERIMENTS_20261013.mmd).
This explanation uses STE guidance. Full ASD-STE100 compliance was not checked.
