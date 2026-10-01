# TPN Subnet Overview (Miner Perspective)

## What you compete on

TPN rewards miners for producing **quantized GGUF versions of a base model** that are small and still accurate. Each competition defines a base model (`model_repo`), a set of benchmarks with minimum scores, and one of two ranking modes:

| `competition_type` | Pass/fail gate | Ranked by |
|---|---|---|
| `benchmark_floor` (default) | Every benchmark ≥ its `min_score` | **Lowest measured RAM** wins |
| `ram_ceiling` | Measured RAM ≤ `max_memory_kb` | **Highest minimum benchmark score** (your weakest benchmark) |

In `ram_ceiling`, a model is only as good as its weakest benchmark — a strong result on one cannot buy rank for a weak result on another.

Competitions are served by the leader validator (`GET /v1/competitions`); `tpn competitions` lists them.

## Competition timeline

Every competition moves through block-based phases:

```
start_block ──OPEN──▶ commit_end_block ──REVEALING──▶ +reveal_grace_blocks ──SCORING──▶ scoring_end_block ──DISTRIBUTING──▶ +distribution_blocks ──COMPLETE
```

- **OPEN** — upload, benchmark, commit.
- **REVEALING** — chain decrypts the timelocked commitments; validators wait out the grace period.
- **SCORING** — the leader verifies, ranks and prechecks submissions.
- **DISTRIBUTING** — validators set weights from the final ranking; emissions flow to winners.
- **COMPLETE** — the competition no longer receives emissions.

## Miner flow

1. **Register** your hotkey on the subnet (`tpn register`).
2. **Upload** your `.gguf` to a **private** HuggingFace repo (`tpn upload`). One repo per submission, exactly one `.gguf` in it. The CLI records repo, filename, **immutable commit SHA** (revision) and the file's **sha256**.
3. **Benchmark** the model yourself on the benchmark service (`tpn benchmark`, paid by you). Every run must be **completed** before the scoring phase starts — a run still in progress then scores 0.0.
4. **Commit** (`tpn commit`): writes a TimeLocked (TLE-encrypted) payload to chain:
   ```json
   {"spec":2,"competition_id":"…","repository":"user/repo","file":"model.gguf",
    "file_sha256":"…","max_memory":<KB>,"huggingface_revision":"<sha>",
    "runs":[{"b":"mmlu","r":"r1234"}, …]}
   ```
   Nobody can read it until the chain auto-decrypts it around `commit_end_block`, so submissions cannot be copied.
5. **Publish** the HF repo (`tpn publish`) before scoring starts. A private repo fails precheck.
6. **Monitor** with `tpn status`, or the leader's `GET /v1/state/competitions/{id}` (per-candidate status and failure reasons).

## Validator side: leader and followers

- One **leader** validator does the real work: reads chain, verifies runs, runs the precheck container, ranks, and publishes results via its API.
- **Followers** download nothing. They poll the leader's final results, compute the same emission weights locally and set weights on chain. When no competition is distributing, every validator copies stake-weighted consensus weights from chain.

## Scoring pipeline (leader)

### Stage 1 — reveal, verify, rank

1. **Scan reveals** from chain `RevealedCommitments`. A reveal is accepted only if it landed within `commit_end_block ± reveal_grace_blocks`, parses as a valid payload, and matches the competition id. **Banned hotkeys are skipped**; unregistered hotkeys are dropped.
2. **Dedup**: if several hotkeys reveal the same `file_sha256`, the **earliest reveal block** wins; the others fail as duplicates. Copying someone's file gains nothing.
3. **Verify runs**: for each competition benchmark, the leader fetches your run id from the benchmark coordinator and accepts it only if:
   - status is `completed`
   - repo matches yours (case-insensitive)
   - it used exactly your committed **revision**
   - your committed **file** is among the run's loaded model files
   - where HF exposes an LFS sha256, it matches your `file_sha256`
   - the run covers that benchmark, the benchmark completed, and it has a score

   Any failure → **that benchmark scores 0.0**. Not fatal by itself, but it will almost certainly fail the floors later. A missing run id also scores 0.0.
4. **Rank** candidates by provisional final score from the verified scores. For `benchmark_floor`, RAM is not yet measured, so this ranking uses your **self-reported** `max_memory`.

### Stage 2 — precheck, in rank order

The leader prechecks **one candidate per loop tick**, top rank first, until `top_n` candidates are scored, candidates run out, or `scoring_end_block` hits. A candidate that fails is marked `failed` and the next one moves up (backfill).

Precheck passes when:
- the HF repo is public
- the committed filename exists at the committed revision
- the **precheck container** (below) accepts the model
- measured RAM is within **±1%** of your reported `max_memory` (`RAM_CHECK_LYING_TOLERANCE`, default 0.01)

After precheck:
- every verified benchmark score must be ≥ its `min_score`
- for `ram_ceiling`: measured RAM must be ≤ `max_memory_kb`

Then the candidate is **scored**:
- `benchmark_floor`: `final_score = −measured_memory_kb` (less RAM ranks higher)
- `ram_ceiling`: `final_score = min(verified benchmark scores)`

### Stage 3 — finalize and set weights

Scored candidates are sorted by `final_score` (descending). Rank *i* receives `emission_distribution[i]` of the competition's pool; ranks past `top_n` get 0.

**Weights only start at `scoring_end_block`**, even if all `top_n` candidates were scored earlier. Finalized results are recorded as soon as scoring completes, but validators only push them on chain once the competition enters the DISTRIBUTING phase, and keep doing so until `scoring_end_block + distribution_blocks`.

While a competition is distributing, each validator:
- sums every distributing competition's shares, scaled by its `emission_weight`
- drops winners no longer registered (their share is not redistributed)
- burns any unallocated weight to uid 0

## The precheck container

A Docker image (`ghcr.io/taofu-labs/tpn-precheck`, built from `shared/validation/precheck.Dockerfile`), public on GHCR. The leader starts one per competition. It runs a small FastAPI service with a pinned CPU build of llama.cpp (`b10020`).

**On startup** it downloads the competition's base model (`model_repo`) once. `/health` reports ready when done.

**`POST /check`** (validator path) downloads your GGUF at the exact committed repo/revision/filename (rejects files > 50 GiB), runs three independent checks, then deletes the file:

1. **Provenance — is this really derived from the base model?**
   - *Tokenizer*: tokenizer type, vocab size and BPE merge count must match exactly; individual entries may differ < 1% (allows added chat-template tokens).
   - *Embedding CKA*: samples 4096 rows of `token_embd.weight` from both models, dequantizes them and computes linear Centered Kernel Alignment. Must be **≥ 0.80**. CKA is invariant to rotation, permutation and scaling, so quantizing or pruning the real base model keeps it high; an unrelated or from-scratch/distilled model falls below the threshold.
   - Fail → `provenance fail`.
2. **RAM measurement**: runs `llama-cli` CPU-only (`-ngl 0`, `-c <ram_check_context_length>` (default 4096), f16 KV cache, `--no-mmap`, generates 1 token) and parses llama.cpp's logged CPU weight and KV-cache buffer sizes. **Measured RAM = weights + KV cache.** Load failure, timeout (600s), OOM kill, or no CPU weight buffer → fail.
3. **sha256** of the downloaded file, compared against your committed `file_sha256`. **A mismatch permanently bans your hotkey** — banned hotkeys are skipped in every future competition.

**`POST /check-local`** (miner self-check) reads a file from a `/data` volume mount and runs the RAM measurement and sha256 (no provenance). Use it to get the exact `max_memory` value before committing:

```bash
docker run -d -p 8080:8080 -v $(pwd):/data ghcr.io/taofu-labs/tpn-precheck
curl -X POST localhost:8080/check-local \
  -H 'content-type: application/json' \
  -d '{"path":"/data/my-model.gguf","context_length":4096}'
# commit ram.ram_bytes / 1024 as max_memory (KB)
```

## Ways to lose (in pipeline order)

| Failure | Consequence |
|---|---|
| Reveal outside grace window / malformed payload / wrong competition id | Not considered |
| Same sha256 as an earlier reveal | Dedup loss |
| Run not completed, or benchmarked a different repo/revision/file | That benchmark = 0.0 |
| Repo private, or file missing at revision | Precheck fail (backfilled) |
| Tokenizer or CKA mismatch with base model | Precheck fail |
| llama-cli can't load the model | Precheck fail |
| Reported `max_memory` off by > 1% from measured RAM | Precheck fail |
| sha256 mismatch | Precheck fail **+ permanent ban** |
| Below a benchmark floor / above the RAM cap | Not scored |
| Ranked below `top_n` | 0 weight |

## Practical tips

- Commit exactly the revision and file you benchmarked; don't push to the repo after benchmarking.
- Measure `max_memory` with `/check-local` at the competition's `ram_check_context_length` — don't guess.
- Make sure every benchmark run has completed before the scoring phase starts.
- Publish the repo before scoring starts.
- In `benchmark_floor`, you're prechecked in order of your *claimed* RAM but paid on *measured* RAM, and a claim off by more than 1% fails outright.
- Emissions only start flowing at `scoring_end_block`, not when your candidate is scored.
