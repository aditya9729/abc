# Full RTC episode for visual inspection

The earlier clip contained only 1.088 seconds of simulation. That was too short
to inspect behavior. The viewer now shows a complete 120.36-second episode.

[Watch the full video](http://127.0.0.1:8770/original-methods/7bbfd1b6f1c87d54.mp4).
[Open the viewer with time and bottle counts](http://127.0.0.1:8770/original-methods/#rtc-heading).

The original RTC evaluator issued 3540 commands on a TRAIN scene. The episode
ended at the time limit with **0/5 bottles in the bin** and no task success.
One bottle was briefly counted during the episode. This is one observation,
not a measured success rate across tasks or seeds.

The policy has one RTC-SFT update and no online RL updates. The evaluator
restored that genuine checkpoint. It used delay 5, replan 8 and prefix
conditioning. The original learner, evaluator and client programs are unchanged.
The explicit visual profile extends the native horizon to 3540 commands and
the author loop bound to 3541. The extra slot observes the terminal state.

Simulation covered 120.36 seconds. The episode took 378.345 seconds of wall time,
including cold inference. Median command spacing was 89.759 ms; p95 was
118.604 ms. The 34 ms control target was not achieved. Video playback follows
simulation time. The viewer reports actual elapsed time separately.

The complete video contains 3541 frames. Its SHA256 is
`7bbfd1b6f1c87d544c0054e1da214d04d025add902941cbecc0145369a061228`.
The readout has one measured progress row per frame, including the initial frame.
It preserves playback when the page refreshes its data.

The evaluator exited with code 0. The native environment and video reported
closed. However, the client process did not exit within the 20-second interrupt
window and required SIGTERM. The canonical job therefore remains **failed**.
The completed episode and video are verified separately. This does not accept
process cleanup. All owned processes are gone and the GPU 0 lease is released.
The 31 checkpoint files remain unchanged.

Run: `author-subset-20261013T193655-d2c6f5`.
Source: `fe258f7`. Evidence is under
`/home/user/aditya/RL/builds/realtime-expo-full-episode-20261013/`:
`ACTUAL_RESULT.json`, `PROGRESS.json`, `BROWSER_RESULT.json`, `ADMISSION.json`,
the original logs, and the recorded configuration. No old clip was overwritten.
A checkpoint directory alias routes new videos to a fresh output parent.
It references the existing checkpoint and creates no additional model copy.

All 39 focused installed checks passed. The new client preserves all 103
installed dependency versions from the previous native client. The optional
package extra now pins its actual `ml-collections==1.1.0` version.
Browser checks verified playback, seeking, the 60-second readout, mobile layout,
and playback continuity across a data refresh. No browser script error occurred.

Root owns execution and publication. The environment boundary handles simulator
calls. The reviewer checks method changes and records. The viewer displays
measured artifacts. [Architecture](REALTIME_AUTHOR_PROGRAM.mmd).
STE guidance used; full ASD-STE100 compliance was not checked.

Online Real-Time EXPO-FT training remains pending. The next work must establish
adequate RTC initialization and resolve client shutdown before another campaign.
