# Reviewed ABC-VLA baseline viewer

The optional viewer presents the approved frozen baseline report. It reads local JSON and does not start policy execution.
At publication, the report has50 complete development episodes,10 history successes and7 native-current successes.
The successful-only mean is935.9 controls. The failure-inclusive mean is987.18 controls.
Learned results and paired measurements remain unavailable.

The accepted source commit is `c453f946b9055eaed3a9dfd30de46f43e0340010`.
Source review and installed checks are separate evidence:

- [Independent source review](vla_ui_review.md) and [pinned review JSON](vla_ui_review.json).
- [Installed wheel and Chromium checks](vla_ui_installed_browser.json). All179 Python files match source, wheel and installation.
- [Changed-report refusal in Chromium](vla_ui_rejected_report_browser.json).
- [Coordinator publication record](vla_ui_publication.json).

The owner passed225 CPU tests. The reviewer passed27 focused tests and27 additional boundary probes.
Two review findings were repaired: incomplete argument pairs cannot overwrite snapshots, and overflowing JSON numbers are rejected.
Installed checks used an isolated environment without policy or simulator libraries.
Desktop and mobile browser checks passed. No external requests or JavaScript errors occurred.
Mobile tables scroll horizontally. The long report pin has a minor visual clipping limitation.

Open http://127.0.0.1:8767 while the coordinator server is active.
The separate offline snapshot is `/home/user/aditya/RL/abc/outputs/bench/progress/qf3-live-20261008/vla-baseline.html`.
Its50 canonical rows include the active capacity run. It is a fixed snapshot, not a live training curve.
Historical snapshots and raw receipts remain unchanged.

The [architecture diagram and team explanation](dashboard.md#optional-abc-vla-development-report) identify the implemented data and control flows.
The producer owns native execution. The reader team validates retained evidence.
The coordinator supplies the approved report pin. The UI agent displays the report.
The independent reviewer checks source behavior. The coordinator owns publication.
This explanation uses STE guidance. A full ASD-STE100 compliance check was not performed.

This milestone delivers presentation software. It does not establish learned improvement, paper reproduction, or R1 Lite performance.
The calibration process uses frozen main source separately from this installed viewer.
Main integration follows the actual calibration exit.
