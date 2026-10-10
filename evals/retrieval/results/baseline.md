# Retrieval baseline — finance-retrieval-curated-v1

Corpus chunks: 47  
Queries: 27

| Strategy | Recall@5 | Recall@10 | MRR@10 | Status |
| --- | ---: | ---: | ---: | --- |
| Dense | — | — | — | NOT MEASURED: local model snapshot unavailable for 'sentence-transformers/all-MiniLM-L6-v2'; download it explicitly before running this strategy |
| BM25 | 0.7593 | 0.8148 | 0.7025 | MEASURED |
| Hybrid RRF | — | — | — | NOT MEASURED: local model snapshot unavailable for 'sentence-transformers/all-MiniLM-L6-v2'; download it explicitly before running this strategy |
| Hybrid RRF + reranker | — | — | — | NOT MEASURED: local model snapshot unavailable for 'sentence-transformers/all-MiniLM-L6-v2'; download it explicitly before running this strategy; local model snapshot unavailable for 'BAAI/bge-reranker-v2-m3'; download it explicitly before running this strategy |

Metrics are macro averages over manual query relevance labels. This small
curated synthetic benchmark is not a production-quality claim.
