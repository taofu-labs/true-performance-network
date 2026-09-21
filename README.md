# TAO Performance Network (TPN)

TPN is a Bittensor subnet that incentivizes miners to build high-performance, quantized language models. Miners benchmark their own GGUF models, then submit the model plus the resulting benchmark run ids via a TimeLocked Commit scheme — hiding the submission until the scoring phase to prevent copying. Validators verify each run actually benchmarked the committed model, rank miners on those verified scores, and check the top candidates for RAM usage, file integrity and base-model provenance before distributing emissions. The subnet runs recurring competitions, each with defined benchmarks, scoring windows, and emission distributions.

## Docs

- [Running a Validator](docs/Validator.md)
- [Miner Operations](docs/Miner.md)
- [Contributor Guide](docs/Contributor.md)
