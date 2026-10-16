# Correct the historical replay order

The new condition is `plus_autonomous_349_before_timeouts`. It uses the same
17,317 real transitions as old B. It changes their insertion order before
sampling starts. The original EXPO sampler, learner, reward and transforms stay
unchanged. Old A and B remain separate conditions with their original behavior.

## Why old B does not test the intended reward signal

Old B appends the 349-control autonomous episode after the 16,968 baseline rows.
Its only positive reward is at row 17,316. The original sampler excludes the
last eight start indices. Its largest eligible start is 17,308. An eight-step
reward window from that start ends at 17,315. No sampled reward window can
include the added terminal reward.

Old B still exposes 341 added success-marked actor starts. It also exposes eight
previous timeout-tail starts. These facts do not make its added terminal reward
sampleable. Preserve its actual training and evaluation results with this limit.
The [tail audit](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/historical-online-fixed-data-tail-review-20261016/TAIL_FINDING.json)
records the original sampler and actual saved-label evidence.

## Minimal assembly change

The new order is:

1. Insert the original 7,069 demonstration rows.
2. Insert the real 349-control autonomous episode with the original `BatchProcessor`.
3. Restore the same 9,899 timeout rows with the original processor.
4. Start the unchanged sampler and ten-call update sequence.

The terminal reward moves to row 7,417. Starts 7,410 through 7,417 now include
that reward. All 349 added actor starts are eligible. The excluded final eight
starts now belong to genuine timeout data. Episode boundaries, masks, success
marks and action-chunk backfill remain owned by the original processor.

No replay array is permuted. No padding, new interaction, second replay copy or
new teacher label is added. Demonstrations remain first, so the existing
measurement banks retain their indices. Equal seeds do not imply equal sampled
batches after this ordering change.

`prepare_before_restore` performs the added insertion only for the new condition.
`prepare_replay` checks normalization, labels and eligible support after restore.
The old two conditions keep their restore-then-append path. The shared update,
prefetch ownership, measurement RNG isolation and full checkpoint saver do not
change.

## Actual CPU evidence and limits

Root accepted [qualification 02](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/historical-online-fixed-data-order-qualification-execution-02/ROOT_ACCEPTANCE.json).
It assembled all 17,317 real rows with the original processor and checked all
349 added rows against accepted normalized evidence. It used the original
sampler on the eight specified positive windows. Returns ranged from
0.9320653677 to 1.0. All eight had bootstrap mask zero and validity one.

This check used capacity 17,318 and diagnostic sample sizes. It did not load a
model or execute B64/UTD20 learning. Forced positive windows establish eligible
support, not stochastic training coverage or task improvement. The terminal next
observation crosses into real timeout data; mask zero prevents bootstrap. It
does not represent a continued environment episode.

The added trajectory still fails the strict placement-footprint check. Its
historical reward stays unchanged. It is not an accepted correction or a new
independent task origin. Corrected-condition training and frozen DEV evaluation
require separate actual receipts.

Six new historical regression cases cover insertion order, reward eligibility,
the old unsampleable tail, unchanged old conditions and refusal before sampling.
The prior twelve historical cases remain unchanged. All 103 installed CPU cases
passed, including the shared driver, existing recovery path and receipt checks.
The package validation also passed after correcting a Path argument in the
release check. No learner change was needed.
[Release evidence](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/historical-online-fixed-data-order-release-01/ROOT_RELEASE.json).
Root owns source/wheel identity, release binding and serial GPU 0 execution. Preserve the
64 GiB disk floor and full-checkpoint save allowance.

The source agent prepares the two helpers and tests. The independent reviewer
checks the frozen change. Root runs qualification and accepts actual results.
See the [architecture](HISTORICAL_ONLINE_FIXED_DATA_ORDER.mmd),
[source review](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/historical-online-fixed-data-order-independent-review-20261016/SOURCE_REVIEW.json)
and [wheel handoff](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/historical-online-fixed-data-order-wheel-preparation-20261016/HANDOFF.json).
This explanation follows STE guidance without a full ASD-STE100 compliance check.
