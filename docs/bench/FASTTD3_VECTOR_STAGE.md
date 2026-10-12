# FastTD3 ABC job supervisor

`abc_bench.fasttd3_vector_stage --config CONFIG.json --output OUTPUT` runs the
original FastTD3 learner against 128 independent ABC CPU worlds. Use the
canonical `abc_bench.runner --algorithm author-subset` launcher. The supervisor
requires its live GPU0 lease before it makes output files or starts children.

The schema1 configuration pins the host configuration, the completed two-world
CPU admission receipt, the executed host payload and the four Fast adapter
payloads. It uses the normal ABC and Fast Python environments. Ready, learner
and shutdown deadlines must fit in the host wall deadline.

The supervisor first checks all 12 Fast author modules, the four protected
function AST hashes and the original dependency versions. It then checks fresh
CPU memory. The estimate is 128 times the largest measured two-world worker
RSS, with 25% margin and 8 GiB extra space. This estimate is not a measured
128-world peak. The host keeps the admitted GPU UUID until it checks the lease,
then hides CUDA before it starts CPU physics. The learner keeps GPU0.

The admitted run has 64 vector calls and 8,192 native transitions. The original
B32768/replay51200/BF16/compiled learner must report 106 critic and 53 actor
updates. The host must report 128 distinct clean worker exits. The two direct
children must exit0. The supervisor checks the original export's step, recipe,
finite actor/critic/normalizer tensors and separate state fingerprints.

This run checks native ABC data collection and original learner capacity.
It does not measure convergence or frozen task success. The export omits
optimizer, replay and RNG state and cannot support exact continuation.
Use a separate explicit phase for longer training and frozen evaluation.

The root agent owns GPU admission and normal release. The Fast adapter agent
owns the public collector and learner boundary. The stage agent owns job
supervision. All children stay in the canonical process group. A failed child,
missing worker, nonfinite export or failed cleanup fails the job. No external
code review is required for this user-requested run.

[Architecture](FASTTD3_VECTOR_STAGE.mmd). This text follows STE guidance;
formal ASD-STE100 compliance was not checked.

CPU tests use explicit child and RPC doubles. Run the tests from outside the
checkout after a normal wheel install, with a short owned pytest `--basetemp`
path. The real 128-world GPU run is a separate root-owned check.
