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

### 4. Commit

Submits a TimeLocked Commit to chain. Auto-reveals at `commit_end_block`.

```bash
tpn commit --wallet <coldkey> --competition <competition-id>
```

Claims (self-reported benchmark scores 0–1) are prompted interactively if not provided. Pass via flag to skip prompts:

```bash
tpn commit -w <coldkey> -c <competition-id> \
  --claims '[{"b":"benchmark-name","s":0.85}]'
```

Use `--dry-run` to inspect the payload without writing to chain.

### 5. Publish repo

Makes the HF repo public so validators can download your model. Do this before the scoring phase begins.

```bash
tpn publish --wallet <coldkey> --competition <competition-id>
```

### 6. Check status

```bash
tpn status --wallet <coldkey>
tpn status --wallet <coldkey> --competition <competition-id>
```

## Local state

The CLI stores per-wallet submission state in:

- Linux/macOS: `~/.tpn/<coldkey>/<hotkey>/<competition-id>.json`
- Windows: `%APPDATA%/tpn/<coldkey>/<hotkey>/<competition-id>.json`

Created on `register`. Each competition file holds the uploaded repo, filename, SHA256, file size, claims, and commit end block. Used by `commit`, `publish`, and `status` to resume without re-entering data.

## Command reference

```
tpn register           Register hotkey on subnet
tpn collateral-status  Show locked miner collateral for a hotkey
tpn competitions       List competitions (--refresh/-r to bypass the 10 min cache)
tpn upload             Upload GGUF to HuggingFace
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
