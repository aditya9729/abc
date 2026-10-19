# Original FastTD3 WBC canonical boundary

This change admits `sadhana_fasttd3_wbc.caller` through the existing `author-subset` runner.
The author learner and canonical lease implementation are unchanged.
The worker checks the real physical GPU UUID lease. It does not require literal `CUDA_VISIBLE_DEVICES=0`.

The source package is `sadhana-fasttd3-wbc`, in the separate Sadhana worktree.
Its `validate_job` checks the actual moving WBC task, R1 Lite embodiment, pinned inputs and finite worker duration.
The canonical loader admits 1,300 seconds for the 12-call qualification, or 28,900 seconds for the 100,000-call reference.
Each limit includes 100 seconds beyond the worker duration. Effective CLI timeouts must preserve that margin.
Other modules retain their existing limits.

The canonical runner supplies a new `fasttd3-wbc` child directory.
The worker writes the standard `receipt.json`; the parent publishes its original process receipt.
The receipt retains the WBC task, embodiment and explicitly supplied scope.
The original runner owns process groups, exclusive GPU admission, cancellation and charging.
The worker samples the declared 10 GiB admission / 8 GiB floor / 2 GiB output limits.
These samples include canonical logs, caches and the short owned temporary directory. They are not a filesystem quota.

Installed preparation is under:
`/home/user/aditya/RL/builds/r1-wbc-chunk-training-20261016/runtime/fasttd3-wbc-native-01`.
Both `qualification/COMMAND.json` and `reference/COMMAND.json` contain exact canonical commands.
No native, model or GPU run has occurred. The 26 worker and 27 canonical CPU tests passed.
They include a synthetic execution double and a real lease check against an isolated synthetic ledger.
The original author module imported without creating a model or CUDA context.

See the Sadhana package README and architecture diagram for action/reward differences and dependency limits.
The legacy `abc-minimal` native dependency metadata conflicts with the borrowed WBC stack.
Only its supervisor modules are used here; no global solver compatibility claim is made.
The original WBC and new worker direct requirements match the installed versions.
No dependency copies, downloads or global requirement changes were made.

The coordinator owns native qualification and result acceptance. The implementation agent prepared this boundary.
An independent reviewer checks the frozen source. CPU tests do not establish an RL improvement.
This explanation uses STE guidance without a full ASD-STE100 compliance check.
