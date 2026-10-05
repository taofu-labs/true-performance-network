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

Integration tests are excluded by default via `addopts = "-m 'not integration'"`,
so `pytest` runs only the fast suite. Opt in with `-m integration`.

## End-to-end competition flow (manual)

`src/validator/tests/test_e2e_flow.py` drives a whole competition — reveal,
run-id verification, ranking, precheck, weights — with the benchmark
coordinator mocked but a **real precheck Docker container**. It downloads a
real GGUF from HuggingFace and measures its RAM with an actual `llama-cli`
run, which is the half the unit suite stubs out.

```bash
# One-off: build the precheck image (add --platform linux/arm64 on Apple silicon)
docker build -t tpn-precheck:test -f shared/validation/precheck.Dockerfile shared/validation

uv run python -m pytest src/validator/tests/test_e2e_flow.py -m integration -v -s
```

Needs Docker and network access; ~70s and a ~400MB model download on first
run (HuggingFace caches it inside the container, which is recreated per test).
Set `PRECHECK_IMAGE` to use a different tag. Tests skip themselves if Docker
or the image is unavailable.

Covered: a verified run scoring and winning; an unverified run (benchmarked a
different repo) scoring 0.0 and losing to an honest lower-scoring miner; a
sha256 mismatch banning the hotkey; and a real RAM measurement above the
competition cap rejecting the candidate.

## Real chain integration (manual)

The localnet setup in `docker/localnet/` + `scripts/setup-localnet.sh` +
`scripts/dev.sh` (see `docs/Contributor.md`) is the way to exercise real
chain behavior end to end — commits, reveals, weight-setting — against a
fast-runtime subtensor. This is not wired into the default pytest suite;
run it manually when you need to validate real chain interaction, not as
part of routine test runs.

`pyproject.toml` registers an `integration` pytest marker ("exercises real
external deps (docker/localnet); run manually, not part of default suite")
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

## Precheck/validation Docker service

`shared/validation/precheck_api.py`'s pure logic (log-parsing regexes,
sha256, the `/check-local` path-traversal guard) is unit-tested directly in
`shared/validation/tests/`. Full end-to-end testing of the service — `docker
build` + `docker run`, a real GGUF download, an actual `llama-cli` invocation
— is deliberately not automated here: it needs no GPU, but a real run can
involve multi-GB model downloads and (if `BASE_MODEL_REPO` is set) up to a
2-hour base-model download before the container reports ready. Treat this as
a manual or future nightly-job concern, not something to run on every push.
