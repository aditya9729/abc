# Original EXPO policy-path diagnostic

The four-start TRAIN check found 2/4 completions with base bypass and 0/4 with
full EXPO or zero edit magnitude. Those runs used different random streams.
They do not isolate selection, editing, or adapter errors. The next check uses
identical recorded observations and explicit base noise before further training.

[Accepted evidence](policy_path_identity_result_20261015.json) records run
`author-subset-20261015T223638-f6d777`, which completed naturally in489.361seconds.
Both fixed-input cases passed the exact route checks. This narrows the diagnosis:
the tested bypass preserves its chosen candidate, while fresh Q selection and
editing deliberately change the command. It does not prove which command succeeds.

All172 saved arrays authenticated and were finite. The30 exact checks comprise
18 repeated preprocessing-leaf checks,10 wrapper/bypass checks, and2 unchanged-bank
checks. They cover two input cases, not30 independent episodes. Actor and head
parameter digests remained equal, and all six optimizer counters stayed zero.

| Measurement | Boot O0 | Prefix O3/O8 |
| --- | --- | --- |
| Fixed candidate through bypass | Bitwise equal | Bitwise equal |
| N1 versus N32 candidate0: largest joint difference | 0.002385rad | 0.001409rad |
| D: selected original candidate, zero-based | 5 | 21 |
| D versus bypass: largest joint difference | 0.098366rad | 0.038626rad |
| E: selected edited candidate, zero-based | 62, from base30 | 50, from base18 |
| E versus bypass: largest joint difference | 0.110640rad | 0.107906rad |
| E versus bypass: largest jaw-width difference | 4.242mm | 4.729mm |

The N1/N32 values use identical candidate-zero noise. They are measured
batch-size arithmetic differences, not evidence of different noise or indexing.
The original base-candidate Q values also differ slightly between D's32-row and
E's64-row critic calls: at most0.00009677 at boot and0.00001755 with prefix.
The diagnostic reports these values rather than asserting cross-batch equality.

The original actor and public wrapper reconstruction match in both cases.
The fresh heads select a different action before learning. The next question is
whether verified outcomes teach useful ranking and recovery. No native action
was executed here, so these differences do not explain the earlier success rates
causally. Historical RNG differences and candidate outcome coverage remain open.

The diagnostic calls the existing admitted author actor, `RTCLearner`, and
`RealTimeEXPOFTLearner` helpers. It changes no learner mathematics, model weights,
replay rule, or production random-number stream. The new `policy_path_probe`
module admits this fixed diagnostic through the existing canonical runner and
child supervisor. It does not replace the original training driver.

| Path | Operation | Comparison |
| --- | --- | --- |
| A | Original actor with one explicit noise sample | Reference arithmetic |
| B | Original full-learner bypass on a fixed bank | Candidate zero, command conversion and execution window |
| C | Original actor with 32 samples, retaining A's noise at index zero | Batch-size arithmetic, measured separately |
| D | Original Q selection on C, without edits | Selected candidate and Q values |
| E | Original Q selection on C plus 32 original edits | Selected candidate, edit and Q values |

The public RTC and full pre-cache methods are also compared with reconstructed
original noise-key chains. This does not restore historical learner RNG state.
Raw unnormalized outputs are retained. Physical comparisons include the original
final JAX float32 conversion. Arm targets use radians; jaw widths use metres.

Inputs are exact TRAIN20044 RGB and state at observations O0, O3 and O8.
The actor loads RTC2000; the full learner initializes fresh heads with seed42.
The boot case uses O0. The prefix case uses O3 for proposals and O8 for selection.
Its five-command prefix comes from reference boot rows3:8. Its execution window
uses rows5:13. The recorded O3/O8 observations did not execute that new prefix.
This is an open-loop comparison, not a simulated continuation.

The result records transformed inputs, noise, candidate banks, Q values,
selected indices, edits, physical commands, parameter digests and update counts.
A completed process can report a discrepancy. Root must inspect that discrepancy.
No tolerance is used to convert batch arithmetic differences into a passing claim.
Saved bfloat16 arrays retain their bytes and declared dtype; NumPy can reload them
as `V2`, which requires a `view(ml_dtypes.bfloat16)` before numerical analysis.

The installed release passed48 CPU boundary and regression checks. All221 Python
files match between source, wheel and installation. Source review found and
corrected a diagnostic comparison across the intentional float64/float32 boundary.
The first source preparation and release remain available. CPU checks establish
neither numerical policy identity nor task success.

Artifacts are under
`/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/`:

- `policy-path-equivalence-caller-preparation-v2-20261015/README.md` describes the fixed caller.
- `policy-path-equivalence-caller-independent-review-20261015/REVIEW.json` records source review.
- `policy-path-equivalence-canonical-release-02/RELEASE.json` records installed qualification.
- `policy-path-equivalence-execution-20261015-01/` contains the job, admission and actual outputs.
- `policy-path-equivalence-execution-20261015-01/ROOT_TENSOR_ANALYSIS.json` records the numerical findings.
- `policy-path-equivalence-execution-20261015-01/ROOT_RESOURCE_CLEANUP.json` records natural exit, closed credit and immutable inputs.

The worker limit is1800seconds. The parent limit is1900seconds. The shared GPU0
lease and existing memory guard remain active. There are no simulator controls
or optimizer updates in this experiment. It cannot establish candidate recovery
coverage, RL improvement, or success on held-out starts.

[Architecture source](POLICY_PATH_IDENTITY_DIAGNOSTIC.mmd) shows the executed
software boundary. Root owns execution and result acceptance. One agent prepared
the caller; a separate agent reviewed it. This text uses STE guidance. Full
ASD-STE100 compliance was not checked.
