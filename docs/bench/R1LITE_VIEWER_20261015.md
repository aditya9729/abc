# R1 Lite results in the original-methods viewer

The accepted local viewer has four cohorts and 80 complete scored videos.
The fifth-cohort extension passed installed CPU and browser-preview checks on 15 October 2026.
It is not published. The new evaluation and its 20 videos remain pending.
Open the current viewer at <http://127.0.0.1:8770/original-methods/#r1lite-heading>.

| CLI cohort | Accepted result | Scope |
| --- | --- | --- |
| `rtc` (default) | 8/20 | RTC-SFT checkpoint2000, containing2,001 updates; failed shutdown remains disclosed. |
| `initialized_full_expo` | 0/20 | Historical full initialization; inherited PRNG settings were not captured. |
| `matched_initialized_full_expo` | 0/20 | Full initialization with declared PRNG settings; zero online updates. |
| `trained_full_expo` | 0/20 | Checkpoint9909;30 policy/editor and600 critic updates. |
| `continued_full_expo` | Pending | One-time approximate restart; a separately accepted frozen evaluation is required. |

The matched pair shows an observed difference of zero percentage points.
One fixed DEV20 cohort does not establish a general RL effect.
The trained9909 producer has incomplete history and separate artifact acceptance.
The saved10808 artifact has complete resumed-segment history, but its producer also failed shutdown.
Its artifact acceptance alone supplies no new policy score or video cohort.
These training and evaluation histories remain separate.

The video-first layout keeps policy selection, episode selection and full video near the top.
Each cohort starts with seed9100 and preserves the original20-episode order.
Refresh retains the chosen episode and playback position when its data is unchanged.
The new label reads policy/critic update counts from accepted evidence.
It labels the checkpoint number separately; checkpoint10808 does not mean10,808 learning updates.

## Publication contract

The installable `abc_bench.report_r1lite_expo` command authenticates root acceptance,
its read-only evidence, source/configuration pins, complete native outcomes, media and sidecars.
Full-policy cohorts require clean completed frozen-evaluator receipts and the exact requested restore.
They retain DEV9100–9119 after unscored startup9001, fullN32/edit32 and delay5.
The matched and continued cohorts also require declared PRNG settings.
Root's actual decoding and visual review remain separate evidence.

The matched-pair branch binds both cohorts to the same producer and comparison configuration.
It retains the parent's incomplete-history restriction.
The continued branch instead requires a root-accepted resumed checkpoint and complete new-segment history.
It checks the actual saved-file inventory, update counters and finite final logged-update metrics.
The parent history stays incomplete. No all-update metric history, exact resume or identical noise is claimed.
Clean frozen evaluation does not convert failed producer supervision into a clean training run.

The publisher holds the existing publication lock and preserves historical result objects.
It copies recordings under content hashes, retains previous results/template backups,
exports the installed template and replaces results atomically.
The continued slot is append-only; a duplicate publication is refused.
It does not replace the four old cohorts or their original matched comparison.
It does not import a learner or start a simulator.

```sh
python -I -B -m abc_bench.report_r1lite_expo \
  --cohort continued_full_expo \
  --acceptance /absolute/actual-frozen-evaluation/ROOT_ACCEPTANCE.json \
  --output-dir /absolute/existing/original-methods \
  --receipt /absolute/fresh/PUBLICATION.json
```

The paths above are placeholders, not accepted future results.
Use the actual frozen-evaluation acceptance only after root completes its checks.
The four earlier cohort options remain supported; omitting `--cohort` still selects RTC.

## Executed verification and remaining work

The previously published four-cohort viewer passed80 HTTP/media-metadata checks.
Root checked selected playback, seeking and refresh in every cohort, plus desktop/mobile layout.
That evidence is in [ROOT_MATCHED_VIEWER_ACCEPTANCE.json](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/initialized-matched-dev20-20261015/ROOT_MATCHED_VIEWER_ACCEPTANCE.json).

The new isolated release passed nine installed publisher tests, including12 new refusal subtests, and19 DOM tests.
Ruff checks and normal wheel installation passed. Installed publisher/template bytes match the qualified source.
The first release attempt stopped at an import-format lint failure before build or tests.
The second release retains that failure and records AST-equivalent formatting corrections.
[CPU qualification](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/continued-viewer-release-20261015-02/CPU_QUALIFICATION.json),
[first attempt](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/continued-viewer-release-20261015-01/QUALIFICATION_STOP.json).

Root then previewed the new installed HTML with the four actual historical cohorts.
Selected playback, seeking, refresh and last-episode selection passed for each cohort.
Desktop/mobile screenshots were captured; the fifth selector stayed disabled and pending.
The browser reported no errors. The live viewer files were unchanged.
This preview does not test a real fifth-cohort score or video.
[Browser preview](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/continued-viewer-release-20261015-02/historical-browser-preview/BROWSER_RESULT.json).

The default CPU tests are `test_report_r1lite_matched.py` and `test_report_r1lite_continued.py`.
Their future checkpoint, receipt and media stand-ins are explicit synthetic fixtures.
`test_r1lite_matched_viewer.cjs` and `test_r1lite_continued_viewer.cjs` check DOM behavior.
The new tests preserve the exact accepted four-cohort JSON fixture and all80 historical media references.
`verify_report_r1lite_initialized.py` remains the unchanged, opt-in historical integration check.
Its earlier seven real-artifact cases are not part of the new nine-test count.

Use the recorded commands in the qualification receipt when a source change needs retesting.
No duplicate suite was run merely to prepare this commit.
After the new evaluation is accepted, root must check the actual100-video publication and all five selectors.
Scientific artifact/evaluation acceptance remains separate from UI qualification.

This commit brings earlier released matched-pair and layout changes into tracked source,
as well as the fifth-cohort extension. It is not only a delta from the four-cohort release.
It changes no learner, simulator, supervisor, runtime dependency or training schedule.
The architecture is [R1LITE_VIEWER_20261015.mmd](R1LITE_VIEWER_20261015.mmd).
Root owns execution and publication. Agents prepare and review source; Leela owns native simulation.
This explanation uses STE guidance; full ASD-STE100 compliance is unchecked.
