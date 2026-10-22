# Miner Operations

## Prerequisites

- Bittensor wallet
- HuggingFace account with write access

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

This quotes everything the competition requires — every benchmark, plus the
**KL check** if the competition configures one — shows the total price against
your available credit, and, once you confirm, submits each run against exactly
the repo, revision and `.gguf` file you uploaded in step 3. Run ids are saved to
your competition config as each one succeeds, ready for `tpn commit`.

The quote table labels each run so you can see what you are paying for:

```
┏━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Run   ┃ Kind      ┃ Price ┃ Notes ┃
┡━━━━━━━╇━━━━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ mmlu  │ benchmark │ $3.20 │ —     │
│ gsm8k │ benchmark │ $3.20 │ —     │
│ kl    │ KL check  │ $5.16 │ —     │
└───────┴───────────┴───────┴───────┘

Total: $11.56  Available credit: $100.00
```

Re-quoting the same model hits the service's identity cache, so iterating on an
unchanged model does not pay twice.

The command returns as soon as each run has an id; it does not wait for the
benchmarks to finish.

| Flag | Effect |
|---|---|
| `--only mmlu,gsm8k` | Run just these (`--only kl` runs the KL check alone) |
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

#### The KL check

If `tpn competitions` shows a **KL check** for the competition, one of your runs
is a *comparison* rather than a benchmark: it measures how far your quantization
moved the model's output distribution away from the competition's base model, on
a private corpus whose text is never published.

That divergence becomes a multiplier on your benchmark score:

```
kl_norm = exp(−KL_p99)
```

A Q4_K_M typically lands around 0.146 nats at P99 → ×0.865. A Q2_K around
1.30 → ×0.27. So a tiny model that scores well on benchmarks but drifts far from
the base is priced down, and accuracy alone is not enough.

**A missing or unverifiable KL run means a multiplier of 0.0, which zeroes your
final score** — and a zero-scoring candidate is not scored at all, so it earns
nothing even if nobody else entered. `tpn benchmark` includes the KL run by
default and `tpn commit` refuses a payload without it, because the commit
payload is sealed — a missing run cannot be added after the fact.

Keep your HuggingFace repo **public** from the moment you benchmark. The
coordinator records whether the repo was private or gated when it read it, and
the validator fails any run that read a non-public repo.

#### Memory

You do not report your model's memory usage. The benchmark service measures the
GPU memory your model takes to load during its runs, and the validator reads
that figure from the run. A competition with a memory cap enforces it against
that measurement.

This is **GPU memory at full offload**, which is not the same quantity as a
host-RAM figure measured on CPU — don't carry a threshold over from one to the
other.

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
the same model you benchmarked. The KL run is checked the same way, plus that it
measured against this competition's base model and corpus.

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
tpn collateral-status  Show locked miner collateral for a hotkey (optional)
tpn competitions       List competitions (--refresh/-r to bypass the 10 min cache)
tpn upload             Upload GGUF to HuggingFace
tpn benchmark          Run the competition's benchmarks + KL check (needs BENCHMARK_API_KEY)
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
(default `https://bench.trueperformancenetwork.com/api/v1`) and
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
