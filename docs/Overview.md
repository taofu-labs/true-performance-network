# TPN Subnet Overview (Miner Perspective)

## What you compete on

TPN rewards miners for producing **quantized GGUF versions of a base model** that are small and still accurate. Each competition defines a base model (`model_repo`), a set of benchmarks with minimum scores, and one of two ranking modes:

| `competition_type` | Pass/fail gate | Ranked by |
|---|---|---|
| `benchmark_floor` (default) | Every benchmark ≥ its `min_score` | **Lowest measured GPU memory** wins |
| `gpu_memory_ceiling` | Measured GPU memory ≤ `max_gpu_memory_kb` | **Highest minimum benchmark score** (your weakest benchmark), times the KL fidelity multiplier |

In `gpu_memory_ceiling`, a model is only as good as its weakest benchmark — a strong result on one cannot buy rank for a weak result on another.

**Memory is measured by the benchmark service**, not by you and not by the validator: it is the GPU memory your model took to load during its runs, reported on every coordinator result. You no longer report a memory figure at all.

Competitions are served by the leader validator (`GET /v1/competitions`); `tpn competitions` lists them.

## Competition timeline

Every competition moves through block-based phases:

```
start_block ──OPEN──▶ commit_end_block ──REVEALING──▶ +reveal_grace_blocks ──SCORING──▶ scoring_end_block ──DISTRIBUTING──▶ +distribution_blocks ──COMPLETE
```

- **OPEN** — upload, benchmark, commit.
- **REVEALING** — chain decrypts the timelocked commitments; validators wait out the grace period.
- **SCORING** — the leader verifies every committed run, then ranks and scores submissions from what those runs report.
- **DISTRIBUTING** — validators set weights from the final ranking; emissions flow to winners.
- **COMPLETE** — the competition no longer receives emissions.

## Miner flow

1. **Register** your hotkey on the subnet (`tpn register`).
2. **Upload** your `.gguf` to a **private** HuggingFace repo (`tpn upload`). One repo per submission, exactly one `.gguf` in it. The CLI records repo, filename, **immutable commit SHA** (revision) and the file's **sha256**.
3. **Benchmark** the model yourself on the benchmark service (`tpn benchmark`, paid by you). Every run must be **completed** before the scoring phase starts — a run still in progress then scores 0.0.
4. **Commit** (`tpn commit`): writes a TimeLocked (TLE-encrypted) payload to chain:
   ```json
   {"spec":3,"competition_id":"…","repository":"user/repo","file":"model.gguf",
    "file_sha256":"…","huggingface_revision":"<sha>",
    "runs":[{"b":"mmlu","r":"r1234"}, …]}
   ```
   Nobody can read it until the chain auto-decrypts it around `commit_end_block`, so submissions cannot be copied.

   A payload committed under spec 2 (which also carried a `max_memory` field) is still accepted, so a commit made just before a validator upgrade is not lost. Spec 1 predates run ids and is rejected.
5. **Publish** the HF repo (`tpn publish`) so the benchmark service can read it, and **leave it public**. The coordinator records whether the repo was private or gated when it read it, and a run against a non-public repo scores 0.0.
6. **Monitor** with `tpn status`, or the leader's `GET /v1/state/competitions/{id}` (per-candidate status and failure reasons).

## Validator side: leader and followers

- One **leader** validator does the real work: reads chain, verifies runs against the benchmark coordinator, ranks, scores, and publishes results via its API. It downloads no models and runs no inference — every number it scores on comes from a coordinator run.
- **Followers** download nothing. They poll the leader's final results, compute the same emission weights locally and set weights on chain. When no competition is distributing, every validator copies stake-weighted consensus weights from chain.

## Scoring pipeline (leader)

Scoring is one stage. The leader verifies every committed run, reads the scores
and the measured memory out of them, ranks, scores and finalizes — all in one
pass, with no model download and no local inference.

### Stage 1 — reveal, verify, rank, score

1. **Scan reveals** from chain `RevealedCommitments`. A reveal is accepted only if it landed within `commit_end_block ± reveal_grace_blocks`, parses as a valid payload, and matches the competition id. Unregistered hotkeys are dropped.
2. **Dedup**: if several hotkeys reveal the same `file_sha256`, the **earliest reveal block** wins; the others fail as duplicates. Copying someone's file gains nothing.
3. **Verify runs**: for each competition benchmark, the leader fetches your run id from the benchmark coordinator and accepts it only if:
   - status is `completed`
   - repo matches yours (case-insensitive)
   - it used exactly your committed **revision**
   - your committed **file** is among the run's loaded model files
   - where HF exposes an LFS sha256, it matches your `file_sha256`
   - the run covers that benchmark, the benchmark completed, and it has a score

   Any failure → **that benchmark scores 0.0**. Not fatal by itself, but it will almost certainly fail the floors. A missing run id also scores 0.0.
4. **Verify the KL run** when the competition sets a `kl_corpus_id` — see *The KL check* below.
5. **Read the measured memory**: the largest GPU figure any *verified* run reported. The KL run is the authoritative source (it loads your model against the competition's base); benchmark runs carry a suite-level figure too, so a competition without a KL check still gets a number. A rejected run's figure is ignored — it describes some other model.
6. **Rank** candidates by final score from the verified scores and the measured memory.
7. **Score**, in rank order, until `top_n` candidates are through. A candidate is scored when:
   - every verified benchmark score is ≥ its `min_score`
   - a verified run actually reported a memory figure (no figure → not scored; a missing number must never read as "fits")
   - for `gpu_memory_ceiling`: measured memory is ≤ `max_gpu_memory_kb`

   A candidate that fails any of these is marked `failed` and the next-ranked one takes its slot (backfill).

   Then:
   - `benchmark_floor`: `final_score = −measured_memory_kb` (less memory ranks higher)
   - `gpu_memory_ceiling`: `final_score = min(verified benchmark scores) × kl_norm`

### Stage 3 — finalize and set weights

Scored candidates are sorted by `final_score` (descending). Rank *i* receives `emission_distribution[i]` of the competition's pool; ranks past `top_n` get 0.

**Weights only start at `scoring_end_block`**, even if all `top_n` candidates were scored earlier. Finalized results are recorded as soon as scoring completes, but validators only push them on chain once the competition enters the DISTRIBUTING phase, and keep doing so until `scoring_end_block + distribution_blocks`.

While a competition is distributing, each validator:
- sums every distributing competition's shares, scaled by its `emission_weight`
- drops winners no longer registered (their share is not redistributed)
- burns any unallocated weight to uid 0

## The KL check

A competition with a `kl_corpus_id` requires one extra run: a **comparison** of your
model against the competition's base model (`model_repo`), measuring how far your
quantization moved the output distribution. Run it with `tpn benchmark` like any
other, and commit its run id under the name `kl`.

The validator accepts the run only if it measured *your* committed artifact against
*this* competition's base on *this* competition's private platform corpus. It then
prices the divergence:

```
kl_norm = exp(−KL_p99)
```

P99 on the `control_to_candidate` pairing, in nats. The control is the base model
converted to F16 on the same pod and run through the same engine, so engine noise
cancels and only quantization damage is left. Measured anchors: a Q4_K_M reaches
0.146 nats at P99 → ×0.865; a Q2_K around 1.30 → ×0.27.

That multiplier scales your benchmark composite, so fidelity and accuracy both
count. **A missing, failed or unverifiable KL run gives `kl_norm = 0.0`**, which
zeroes the final score — an unverifiable run must never read as perfect fidelity.
A candidate scoring 0 is not scored at all, so it cannot occupy a payable rank
even when the field is thin. Nothing about a failed KL check bans a hotkey.

`benchmark_floor` competitions ignore the multiplier: that mode ranks on negated
memory, which a (0,1] factor would *improve*.

## Ways to lose (in pipeline order)

| Failure | Consequence |
|---|---|
| Reveal outside grace window / malformed payload / wrong competition id | Not considered |
| Same sha256 as an earlier reveal | Dedup loss |
| Run not completed, or benchmarked a different repo/revision/file | That benchmark = 0.0 |
| Committed sha256 contradicts the hash the coordinator reports | That run = 0.0 |
| Repo private or gated when the run read it | That run = 0.0 |
| KL run missing, failed, or measured against the wrong base or corpus | `kl_norm = 0.0` → **not scored**, no emission |
| No verified run reported a memory figure | Not scored (backfilled) |
| Below a benchmark floor / above the memory cap | Not scored (backfilled) |
| Ranked below `top_n` | 0 weight |

## Practical tips

- Commit exactly the revision and file you benchmarked; don't push to the repo after benchmarking.
- Make sure every run has completed before the scoring phase starts, including the KL run.
- Publish the repo before you benchmark, and leave it public — a run that read a private or gated repo scores 0.0.
- You no longer report a memory figure — it is measured during your runs. Memory is **GPU memory at full offload**, so it is not comparable to a CPU-side RAM number you may have measured before.
- If the competition requires a KL check, commit its run id. Without it your final score is 0, and the payload is sealed at commit so it cannot be added later.
- Emissions only start flowing at `scoring_end_block`, not when your candidate is scored.
