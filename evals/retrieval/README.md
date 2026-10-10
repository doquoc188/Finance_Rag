# Curated finance retrieval benchmark v1

This directory contains a small synthetic/curated benchmark for retrieval architecture
regressions. It is not sampled from private reports and does not represent all production
financial documents, languages, companies, or question types. Its financial statements
and values are fixtures, not claims about the named companies.

## Data files

`corpus.jsonl` has 47 chunks across FPT, VNM, HPG, MWG, VCB, and SSI. Every record has an
explicit `chunk_id`, `document_id`, text, and scalar metadata. IDs beginning with `bench_`
are benchmark-owned and do not use the production order-dependent ID generator.

`queries.jsonl` has 27 questions. Every record contains a unique `query_id`, nonblank
question, and one or more `relevant_chunk_ids`; category and language are descriptive.
The set contains 11 Vietnamese, 13 English, and 3 mixed-language questions:

| Category | Queries |
| --- | ---: |
| bilingual terminology | 2 |
| exact company/metric | 6 |
| hard metric | 4 |
| hard period | 2 |
| hard year | 6 |
| multi-relevant | 2 |
| semantic paraphrase | 5 |

Nearby years, adjacent finance metrics, quarterly versus annual periods, percentages,
Vietnamese `tỷ`/`triệu`, English units, and repeated company tokens provide distractors.
The two multi-relevant questions require two chunks. Relevance IDs were assigned manually
while curating the fixture. Dense, BM25, RRF, BGE, and LLM output did not create the gold.
Load-time validation rejects duplicate IDs, blank questions, empty/duplicate gold sets,
dangling relevant IDs, malformed JSON, unexpected fields, and invalid metadata.

## Metrics

For each question:

```text
Recall@K = unique relevant IDs retrieved in the first K / all relevant IDs
MRR@K    = 1 / rank of the first relevant result in the first K, else 0
```

The reported Recall@5, Recall@10, and MRR@10 values are arithmetic macro averages across
all 27 questions. Internal JSON keeps full floating-point values. Markdown rounds to four
decimal places for display.

## Strategies and fixed configuration

All strategies use this exact corpus, query order, and gold labels. Stable labels are:

- `dense`: current `QdrantDenseRetriever`, in-memory cosine Qdrant, and the configured
  local MiniLM embedding model.
- `bm25`: current `BM25Retriever`.
- `hybrid_rrf`: dense top 20 plus BM25 top 20, RRF (`k=60`) top 20, final top 10.
- `hybrid_rrf_reranked`: the same fused top 20 followed by the Phase 1.3B BGE adapter,
  final top 10.

The runner always asks each evaluated strategy for ten results, so production
`final_top_k=4` cannot truncate Recall@10 or MRR@10. These values are the initial fixed
baseline configuration and were not tuned against the benchmark.

## Offline execution and outputs

Run all four requested labels from the repository root:

```powershell
uv run python -m evals.retrieval.run_benchmark
```

The runner forces Hugging Face/Transformers offline mode and resolves model snapshots with
`local_files_only=True`. It never calls OpenAI or remote Qdrant and never substitutes the
legacy constant-vector embedding fallback. Available strategies are measured; unavailable
models produce explicit `NOT MEASURED` entries while other strategies continue.

Current local execution measured actual BM25 only because neither
`sentence-transformers/all-MiniLM-L6-v2` nor `BAAI/bge-reranker-v2-m3` was cached. Results
are in `results/baseline.json` and `results/baseline.md`. Actual BM25 measured:

| Strategy | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| BM25 | 0.7593 | 0.8148 | 0.7025 |

These are measurements on this 47-chunk curated fixture, not production superiority or a
general accuracy claim. Dense, Hybrid RRF, and reranked metrics are absent rather than
filled with fake scores.

After both models are already available locally, run the same fixed configuration into a
new result file. Local directories can be supplied explicitly without a download:

```powershell
uv run python -m evals.retrieval.run_benchmark --embedding-model "D:\models\all-MiniLM-L6-v2" --reranker-model "D:\models\bge-reranker-v2-m3" --json-output evals/retrieval/results/local-full.json --markdown-output evals/retrieval/results/local-full.md
```

Use `--strategies bm25` for a BM25-only run and `--json-output` /
`--markdown-output` to preserve another result set. Do not overwrite this baseline after
tuning; create a separately named, versioned experiment instead.

Pytest uses deterministic fakes to verify metric/runner behavior and the actual BM25 core
for a smoke path. Fake scores are test evidence only and are never written as measured
model results. The real embedding and BGE models are not loaded by normal tests.
