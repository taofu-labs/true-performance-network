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

## Real chain integration (manual)

The localnet setup in `docker/localnet/` + `scripts/setup-localnet.sh` +
`scripts/dev.sh` (see `docs/Contributor.md`) is the way to exercise real
chain behavior end to end — commits, reveals, weight-setting — against a
fast-runtime subtensor. This is not wired into the pytest suite; run it
manually when you need to validate real chain interaction, not as part of
routine test runs.

## Precheck/validation Docker service

`shared/validation/precheck_api.py`'s pure logic (log-parsing regexes,
sha256, the `/check-local` path-traversal guard) is unit-tested directly in
`shared/validation/tests/`. Full end-to-end testing of the service — `docker
build` + `docker run`, a real GGUF download, an actual `llama-cli` invocation
— is deliberately not automated here: it needs no GPU, but a real run can
involve multi-GB model downloads and (if `BASE_MODEL_REPO` is set) up to a
2-hour base-model download before the container reports ready. Treat this as
a manual or future nightly-job concern, not something to run on every push.
