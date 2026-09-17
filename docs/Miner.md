# Miner Operations

## Prerequisites

- Bittensor wallet
- HuggingFace account with write access
- Miner collateral: currently at least `0.3 TAO` locked on the submitting hotkey
  for each competition you enter

## Install CLI

Installs `uv` if not present, then installs the `tpn` CLI tool:

```bash
./install_cli.sh
```

Verify:
```bash
tpn --help
```

## Workflow

### 1. Register on subnet

```bash
tpn register --coldkey <coldkey> --hotkey default
```

The registration flow can also lock miner collateral and set a collateral floor.
For the current live requirement, use at least `0.3 TAO`:

```bash
tpn register \
  --coldkey <coldkey> \
  --hotkey default \
  --collateral-amount 0.3 \
  --floor-amount 0.3
```

If the hotkey is already registered, make sure it has at least `0.3 TAO` locked
before the competition scoring phase starts. Validators check collateral at
scoring time. A hotkey below the threshold is skipped for that competition; it is
not banned for being under collateralized.

Check the current collateral position:

```bash
tpn collateral-status --wallet <coldkey> --hotkey default
```

### 2. List competitions

```bash
tpn competitions
tpn competitions --all   # include inactive
```

### 3. Upload model to HuggingFace

Uploads your `.gguf` file to a private HF repo and saves upload metadata locally.
Use one HuggingFace repo per submission, and keep exactly one `.gguf` file in
that repo so validators test the intended file.

```bash
tpn upload ./my-model.gguf \
  --repo my-model-q4 \
  --coldkey <coldkey> \
  --competition <competition-id>
```

### 4. Run the benchmarks

You run your own benchmarks before committing, and pay for them yourself.

First, create an API key on the benchmark service: log in to its web GUI, add
credits, then create a key on the Tokens page. API keys can spend credits but
cannot buy them, so top up in the GUI first.

```bash
export BENCHMARK_API_KEY=bapi_...

tpn benchmark --wallet <coldkey> --competition <competition-id>
```

This quotes every benchmark the competition requires, shows the total price
against your available credit, and — once you confirm — submits each run
against exactly the repo, revision and `.gguf` file you uploaded in step 3. Run
ids are saved to your competition config as each one succeeds, ready for
`tpn commit`.

The command returns as soon as each run has an id; it does not wait for the
benchmarks to finish.

| Flag | Effect |
|---|---|
| `--only mmlu,gsm8k` | Run just these benchmarks |
| `--rerun` | Re-run benchmarks that already have a saved run id |
| `--yes` | Skip the price confirmation prompt |
| `--force-price` | Accept the current price without quoting first |

Re-running one benchmark is cheap: `tpn benchmark --only <name> --rerun`
replaces just that run id and leaves the others alone. Runs of the same model
from an earlier competition stay valid, so an unchanged model needs no
re-benchmarking.

A run must have **completed** before the competition's scoring phase starts.
There is no waiting period — a run still in progress at that point scores 0.0
for that benchmark. Check progress in the benchmark service GUI.

Driving the benchmark service directly is also fine: `tpn commit --runs`
accepts run ids from any source.

### 5. Commit

Submits a TimeLocked Commit to chain. Auto-reveals at `commit_end_block`.

```bash
tpn commit --wallet <coldkey> --competition <competition-id>
```

`tpn benchmark` already saved your run ids, so this normally needs no extra
input. To supply them yourself:

```bash
tpn commit -w <coldkey> -c <competition-id> \
  --runs '[{"b":"mmlu","r":"r1234"},{"b":"hellaswag","r":"r1235"}]'
```

A run id is either the service's short id (`r1234`) or the full UUID.

Validators verify every run id against your commit: the run must have
benchmarked the same repo, the same revision and the same file you committed.
A run that benchmarked anything else scores 0.0 for that benchmark, so commit
the same model you benchmarked.

Use `--dry-run` to inspect the payload without writing to chain.

### 6. Publish repo

Makes the HF repo public so validators can download your model. Do this before the scoring phase begins.

```bash
tpn publish --wallet <coldkey> --competition <competition-id>
```

### 7. Check status

```bash
tpn status --wallet <coldkey>
tpn status --wallet <coldkey> --competition <competition-id>
```

## Local state

The CLI stores per-wallet submission state in:

- Linux/macOS: `~/.tpn/<coldkey>/<hotkey>/<competition-id>.json`
- Windows: `%APPDATA%/tpn/<coldkey>/<hotkey>/<competition-id>.json`

Created on `register`. Each competition file holds the uploaded repo, filename, SHA256, file size, benchmark run ids, and commit end block. Used by `commit`, `publish`, and `status` to resume without re-entering data.

## Command reference

```
tpn register           Register hotkey on subnet
tpn collateral-status  Show locked miner collateral for a hotkey
tpn competitions       List competitions (--refresh/-r to bypass the 10 min cache)
tpn upload             Upload GGUF to HuggingFace
tpn benchmark          Run the competition's benchmarks (needs BENCHMARK_API_KEY)
tpn commit             Submit TimeLocked Commit to chain
tpn publish            Make HF repo public
tpn status             Show submission state
tpn version            Print CLI version
```

## Overriding defaults

For local development or non-standard setups, all commands accept these global flags before the subcommand:

| Flag | Default | Description |
|---|---|---|
| `--network` | `finney` | Chain endpoint |
| `--netuid` | `65` | Subnet UID |
| `--wallet-path` | `~/.bittensor/wallets` | Override wallet directory |
| `--leader-url` | `https://val0.trueperformancenetwork.com` | Override leader validator API (serves competition configs) |
| `--block-time` | `12.0` | Seconds per block (use `0.300` for localnet) |

`tpn benchmark` additionally takes `--billing-url`
(default `https://benchmarks.trueperformancenetwork.com/api/v1`) and
`--api-key` (`$BENCHMARK_API_KEY`) after the subcommand.

Example:
```bash
uv run --package cli tpn \
  --network ws://localhost:9946 \
  --netuid 2 \
  --block-time 0.300 \
  --leader-url http://localhost:9200 \
  --wallet-path ./wallets \
  commit -w charlie -c tpn-localnet
```
