# Catcher: original policy and adapter action check

The adapter matches the original frozen policy on one recorded observation and
pending action prefix. The largest full-chunk difference is **1.79 × 10⁻⁷**.
The released `RTCLearner` returns the expected four-action postfix exactly.
This checks an inference boundary. It does not establish a learning gain or
equivalence between complete evaluation runs.

## Actual inputs and execution

The inputs come from the accepted BC24 trial at seed100001 in the
[four-start frozen comparison](REALTIME_CATCHER_FROZEN_20261014.md).
The observation has 679 values. The pending prefix has four actions with six
values each. Both hashes match the first recorded asynchronous submission.
The four issued commands fall between its submission and consumption timestamps.

Root ran two CPU phases using existing, normally installed packages:

1. The original RTC `FlowPolicy` restored all 42 BC24 parameter leaves. It
   produced boot and guided chunks with eight actions each.
2. The Catcher backend consumed the same observation, prefix and Gaussian draws.
   The unmodified EXPO `RTCLearner` also sampled through its normal interface.

Both phases used ten Euler steps, delay 4, execution horizon 4, exponential
prefix weights and maximum guidance 5. The original model and `RTCLearner`
have no supplied-noise argument. The backend's optional noise argument is an
explicit integration change. The diagnostic exported original draws and
verified the learner/backend random-key mapping before comparing outputs.
It did not replace either sampler or patch its random generator.

The original runtime uses JAX 0.4.35 and jaxlib 0.4.34. The backend runtime uses
JAX/jaxlib 0.5.3. Both use Flax 0.10.2 and the legacy Threefry setting. Gaussian
draws and random keys match exactly across these CPU runtimes.

## Measured differences

| Comparison | Maximum absolute difference |
| --- | ---: |
| Original versus backend, full boot chunk | 1.1920929e-7 |
| Original versus backend, full guided chunk | 1.7881393e-7 |
| Backend versus released RTC boot output | 0 |
| Backend positions 4:8 versus released RTC guided output | 0 |
| Original positions 4:8 versus released RTC guided output | 1.1920929e-7 |
| Returned `executed_padded` slices versus expected actions | 0 |

The declared output tolerance was `atol=2e-5, rtol=2e-5` before execution.
Changing the supplied boot noise changed the output by up to 1.93957698. Thus,
the tested noise input affects sampling. Parameters, optimizer state, EMA state
and step zero remain unchanged. The learner RNG advances as expected.

As a separate diagnostic, regenerated boot commands differ from the historical
first four issued commands by at most 3.88e-7. The guided postfix differs from
the next four historical commands by at most 3.28e-7. These values do not provide
a new native trajectory or outcome.

The original phase completed in 11.06 seconds. The backend phase completed
in 48.02 seconds, including imports and compilation. Both exited 0 with empty
owned process groups and no termination escalation. These durations are not
steady-state inference latency. No simulator, GPU job or training update ran.
All protected source and checkpoint pins remained unchanged.

## What this resolves

There is no observed action-conversion or postfix-slicing defect at this
recorded boundary. This finding narrows the investigation. It does not prove
all adapter behavior, GPU execution or long trajectories correct.

The [100-stream original-evaluator results](CATCHER_READINESS_20261014.md)
remain separate from the four-start EXPO-client results. The drivers differ
in batching, random-key paths, numerical runtime and control orchestration.
This diagnostic does not align all those factors or transfer a success rate
between them.

The earlier R2 run made 34 actor calls within a 1,000-call warmup. It does not
test mature learning. The prepared next profile uses the original transition-
derived update schedule and a declared full-prior mixture. Its GPU execution
and later frozen evaluation remain pending.

## Evidence and ownership

[Evidence index](CATCHER_INFERENCE_PARITY_20261014.json) pins the scripts,
inputs, actual outputs and root acceptance. Raw artifacts are under
`/home/user/aditya/RL/builds/realtime-catcher-inference-equivalence-20261014/`.
Six input-association fixtures passed before model execution. Root separately
read the saved arrays and verified the measured errors and exact output slices.

[Architecture](CATCHER_INFERENCE_PARITY_20261014.mmd). The author owns the model
and RTC sampler. The adapter supplies the model interface. The preparation
agent selects recorded inputs and comparison checks. Root executes the two
phases and accepts the evidence. A separate agent reviews the result.
The explanation uses STE guidance. Full ASD-STE100 compliance was not checked.
