# Optional ResFiT profile transport

The existing canonical CLI accepts `--resfit-profile` only with `--algorithm resfit-abc-vla`. Default worker commands, learning configuration, parent watchdog, original GPU0 lease, ledger reservation and accounting stay unchanged. The option selects one fixed source-pinned public NRH launcher file. Output is `<invocation>/profile`, separate from `<invocation>/training`.

After child return or execution error, `receipt.profile.complete` reports export completeness separately from the existing worker/training status and exit. Missing output, an explicit export failure, size/hash mismatch or malformed pstats reports false. A worker return 2 remains 2 even if profiling fails. Completed training does not imply a usable profile. Root admission must require profile completeness when admitting a cost measurement.

Future command, **not launched or admitted**:

```sh
cd /home/user/aditya/RL/abc-resfit-storage-diagnostic
.venv/bin/python -m abc_bench.runner --algorithm resfit-abc-vla --training-config ABS_REVIEWED_CONFIG --timeout-seconds ROOT_ADMITTED_SECONDS --resfit-profile
```

The normal editable full ABC distribution resolves this controller in its existing private runtime. No environment rebuild is required. Canonical outputs/bench resolves to `/home/user/aditya/RL/abc/outputs/bench`; the original lock and ledger remain authoritative. The NRH worker's ten source identities and native model assets are unchanged. Review/admission precedes native use.

CPU evidence and exact commands are in `/home/user/aditya/RL/builds/resfit-native-profiler-implementation-20261009`. Unit worker doubles are fixtures. The actual installed-worker `--resolve` smoke returned zero and `native_constructed:false`, with ten prior source identities unchanged and a valid profile. It did not construct a model/world or measure native costs. See the owning NRH `harness/docs/RESFIT_PROFILE.md` for architecture and STE-guided roles.
