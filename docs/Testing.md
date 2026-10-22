# Testing

## Fast unit suite

```bash
uv sync
uv run python -m pytest
```

(Use `python -m pytest`, not bare `pytest` — on machines with a pyenv/global
`pytest` shim on `PATH`, `uv run pytest` can resolve that shim instead of the
workspace `.venv`'s pytest.)

Everything under `*/tests/` runs against mocks/fakes — no network, no chain,
no Docker. Bittensor is faked at the `common.chain` seam (`get_subtensor`,
`get_wallet`, etc.) or via constructor injection on `Validator`
(`subtensor=`, `metagraph=`, `wallet=`). The benchmark coordinator uses the
shipped `MockCoordinator` (`competition.benchmark_client`) instead of a real
HTTP call.

`addopts = "-m 'not integration'"` excludes tests tagged `integration`.
Nothing carries that marker today — removing the precheck container left the
whole suite runnable in-process — so `pytest` runs everything.

## End-to-end competition flow

`src/validator/tests/test_e2e_flow.py` drives a whole competition — reveal,
run-id verification, ranking, scoring, weights — with only the benchmark
coordinator mocked. The validator downloads nothing and runs no inference, so
this needs no Docker, no network and no GPU, and runs in the default suite.

```bash
uv run python -m pytest src/validator/tests/test_e2e_flow.py -v
```

Covered: a verified run scoring and winning; an unverified run (benchmarked a
different repo) scoring 0.0 and losing to an honest lower-scoring miner; a
sha256 that contradicts the coordinator's hash scoring 0 without banning; a
measured memory figure above the competition cap rejecting the candidate; and
a run reporting no memory figure at all not being scored.

## Real chain integration (manual)

The localnet setup in `docker/localnet/` + `scripts/setup-localnet.sh` +
`scripts/dev.sh` (see `docs/Contributor.md`) is the way to exercise real
chain behavior end to end — commits, reveals, weight-setting — against a
fast-runtime subtensor. This is not wired into the default pytest suite;
run it manually when you need to validate real chain interaction, not as
part of routine test runs.

`pyproject.toml` registers an `integration` pytest marker ("exercises real
external deps (localnet); run manually, not part of default suite")
for tests that need this kind of real dependency — use it to tag any test
that shouldn't run in the fast unit suite.

### Mocked benchmark runs on localnet

Miners own their run ids in the live flow — the validator only polls them — so
`MockCoordinator` reports any id it did not generate itself as
`unknown run_id`. To drive a full localnet competition with benchmarks mocked
and everything else real, pre-register the ids in a JSON file and point
`MOCK_RUNS_FILE` at it:

```json
{
  "r1001": {
    "repo": "owner/model-GGUF",
    "revision": "<immutable commit sha>",
    "file": "model-q4.gguf",
    "file_sha256": "<64 hex>",
    "benchmark": "mmlu",
    "score": 0.82
  }
}
```

```bash
MOCK_RUNS_FILE=./mock_runs.json TPN_DOTENV_PATH=.env.localnet \
  uv run --package validator python src/validator/main.py
```

`repo`, `revision` and `file` must match what the miner commits, or the run is
rejected exactly as a real mismatch would be. An id absent from the file still
fails, so verification is never weakened. Only `MockCoordinator` reads this —
`BENCHMARK_BACKEND=http` ignores it.

### Commit reveals on a fast-runtime chain

TLE reveals are scheduled in drand rounds (wall-clock), but `scan_reveals`
accepts them by block height, within `reveal_grace_blocks` of
`commit_end_block`. The two only line up while `--block-time` matches the
chain's real rate.

Localnet's fast runtime drifts enough for this to matter: a commit placed far
ahead of `commit_end_block` can auto-decrypt a hundred blocks early and get
discarded, ending the competition with `failed_no_reveals` despite a perfectly
valid payload. Mainnet holds close to 12s, so the default 150-block grace
absorbs it there.

When testing on localnet: measure the real block rate, pass it as
`--block-time`, keep the commit window short, and give the competition a
generous `reveal_grace_blocks`. If reveals are being dropped, the validator
logs the reason at DEBUG:

    reveal_block=1017 outside window for commit_end_block=1133

## Coordinator payload shapes

The validator reads benchmark scores, the KL divergence and the measured GPU
memory out of coordinator `/status` bodies, so the parser is pinned against a
real captured payload in `src/validator/tests/fixtures/comparison_status.json`
rather than a hand-written shape. That capture predates the server's `vram`
block, so the memory parsing is covered by a clearly-labelled synthetic case
(`test_parses_the_vram_block_a_comparison_reports`) built from the server's
own projection code. Re-capture the fixture from a current run when
convenient, and the synthetic case can go.
