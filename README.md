# TAO Performance Network (TPN)

TPN is a Bittensor subnet that incentivizes miners to build high-performance, quantized language models. Miners submit GGUF models via a TimeLocked Commit scheme — hiding their submission until the scoring phase to prevent copying. Validators download and benchmark each model, then distribute emissions based on performance and efficiency. The subnet runs recurring competitions, each with defined benchmarks, scoring windows, and emission distributions.

## Miner collateral

Miners currently need collateral to be eligible for scoring. For each competition
you enter, the submitting hotkey should have at least `0.3 TAO` locked as miner
collateral before scoring starts.

Validators check this at scoring time. A hotkey below the collateral threshold is
skipped for that competition; it is not banned for being under collateralized.

When registering with the TPN CLI, you can lock collateral and set a collateral
floor during registration. Existing miners can check their current position with:

```bash
tpn collateral-status --wallet <coldkey> --hotkey <hotkey>
```

## Docs

- [Running a Validator](docs/Validator.md)
- [Miner Operations](docs/Miner.md)
- [Contributor Guide](docs/Contributor.md)
