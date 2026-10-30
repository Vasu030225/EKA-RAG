# RAG POC Evaluation Report

## Run summary

- **Run date (UTC):** 2026-10-30 05:19:21
- **Platform:** Windows-11-10.0.26200-SP0
- **Python:** 3.14.6
- **LLM:** Ollama `llama3.2` at `http://localhost:11434`
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Source documents/pages/chunks:** 1 / 4 / 8
- **FAISS index size:** 0.02 MiB (temporary evaluation index)
- **Question repetitions:** 3
- **Functional checks:** 5/5 passed

## Test results

Checks require expected key terms or the exact unknown-answer fallback, plus the expected cited page where specified. These are sample-document regression checks, not a general measure of answer quality.

| Test | Result | Cited page(s) | Mean latency (s) | Error |
|---|---|---:|---:|---|
| Annual leave entitlement | PASS | 1, 2 | 1.376 | - |
| Annual leave carry-forward | PASS | 1, 2, 3 | 9.707 | - |
| Work-from-home allowance | PASS | 1, 2, 3 | 8.977 | - |
| Password requirement | PASS | 2, 4 | 12.242 | - |
| Unsupported question | PASS | 1, 2, 3, 4 | 12.973 | - |

## Timing

| Measurement | Seconds |
|---|---:|
| PDF loading and chunking | included in index build |
| FAISS build (PDF load, chunking, model initialization/embedding, save) | 122.181 |
| Saved FAISS index load | 2.002 |
| First warm-up question | 34.684 |
| Mean question latency (warm-up excluded) | 9.055 |
| Median question latency | 2.903 |
| Approx. p95 question latency | 33.565 |

Warm-up query: 34.684 seconds.

## Cost effectiveness

- **Hosted API charges:** INR 0 for this local Ollama run. The embedding model and LLM execute locally; no hosted inference API is called.
- **Estimated index-build electricity:** INR 0.027151 (0.00339392 kWh).
- **Estimated electricity per question:** INR 0.00201226, using 100 W and INR 8/kWh.
- **Estimated total per question with index-build amortized over 1000 requests:** INR 0.00203941.
- **Comparison:** No hosted API rate was provided; a cloud-cost comparison is intentionally not estimated.

Electricity is an estimate based on user-supplied device power and energy rate, not a wall-power measurement. Results exclude hardware purchase/depreciation, cooling, idle time, and one-time model downloads. The local approach avoids per-request hosted API fees but shifts costs to local compute and model storage.

## Reproduce

```powershell
python -m scripts.evaluate_poc
```

Tune assumptions, set an optional cloud baseline, or choose another report path:

```powershell
python -m scripts.evaluate_poc --repeats 3 --power-watts 100 --electricity-rate-inr-per-kwh 8 --cloud-cost-per-1000-requests-inr 50 --output reports/evaluation_report.md
```
