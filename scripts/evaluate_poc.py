"""Run grounded sample questions and write a latency and local-cost report."""

import argparse
import math
import platform
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "evaluation_report.md"
UNKNOWN_ANSWER = "I don't know based on the provided documents."


@dataclass(frozen=True)
class EvaluationCase:
    """A sample question and simple, deterministic checks for its result."""

    name: str
    question: str
    required_terms: tuple[str, ...] = ()
    expected_page: int | None = None
    expected_exact_answer: str | None = None


CASES = (
    EvaluationCase(
        "Annual leave entitlement",
        "How many days of paid annual leave does a full-time employee get per year?",
        ("24",),
        expected_page=1,
    ),
    EvaluationCase(
        "Annual leave carry-forward",
        "How many unused annual leave days can an employee carry forward?",
        ("10",),
        expected_page=1,
    ),
    EvaluationCase(
        "Work-from-home allowance",
        "What is the monthly work from home allowance?",
        ("1000",),
        expected_page=2,
    ),
    EvaluationCase(
        "Password requirement",
        "What is the minimum password length?",
        ("12",),
        expected_page=4,
    ),
    EvaluationCase(
        "Unsupported question",
        "What was Acme's 2024 revenue?",
        expected_exact_answer=UNKNOWN_ANSWER,
    ),
)


def _normalized(text: str) -> str:
    """Normalize spaces and punctuation separators for numeric answer checks."""
    return "".join(character.lower() for character in text if character.isalnum())


def _matches_case(answer: str, pages: list[int], case: EvaluationCase) -> bool:
    if case.expected_exact_answer is not None:
        answer_matches = answer.strip() == case.expected_exact_answer
    else:
        normalized_answer = _normalized(answer)
        answer_matches = all(
            _normalized(term) in normalized_answer for term in case.required_terms
        )
    page_matches = case.expected_page is None or case.expected_page in pages
    return answer_matches and page_matches


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}"


def _write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _blocked_report(error: ImportError, args: argparse.Namespace) -> str:
    rows = "\n".join(
        f"| {case.name} | {case.question} | NOT RUN | {error.name or str(error)} |"
        for case in CASES
    )
    return f"""# RAG POC Evaluation Report

## Run status

**Live evaluation not run:** a required Python dependency could not be imported: `{error.name or str(error)}`.

Install the project dependencies with `python -m pip install -r requirements.txt`, then rerun:

```powershell
python -m scripts.evaluate_poc
```

## Test cases

| Test | Question | Status | Blocker |
|---|---|---|---|
{rows}

## Time and cost effectiveness

No application latency or energy measurements were collected because the RAG pipeline could not start. Local Ollama API charges are **0**; this does not mean total operating cost is zero. A completed run estimates electricity from measured execution time and the configured assumptions: **{args.power_watts:g} W** device power and **INR {args.electricity_rate_inr_per_kwh:g}/kWh**. Hardware, cooling, model downloads, and idle power are excluded.

To compare against a hosted API, rerun with `--cloud-cost-per-1000-requests-inr <your-price>`; no provider price is assumed here.
"""


def run_evaluation(args: argparse.Namespace) -> str:
    try:
        from src.chunker import split_documents
        from src.config import load_settings
        from src.loader import load_pdf_documents
        from src.rag_chain import answer_question
        from src.vector_store import build_index, load_index
    except ImportError as exc:
        return _blocked_report(exc, args)

    settings = load_settings()
    started = time.perf_counter()
    documents = load_pdf_documents(settings.data_dir)
    chunks = split_documents(documents, settings)
    with tempfile.TemporaryDirectory(prefix="eka-rag-eval-") as temp_dir:
        evaluation_settings = replace(settings, index_dir=Path(temp_dir) / "faiss_index")
        vector_store = build_index(chunks, evaluation_settings)
        build_seconds = time.perf_counter() - started

        load_started = time.perf_counter()
        vector_store = load_index(evaluation_settings)
        load_seconds = time.perf_counter() - load_started
        index_bytes = sum(
            path.stat().st_size
            for path in evaluation_settings.index_dir.iterdir()
            if path.is_file()
        )

        warmup_started = time.perf_counter()
        warmup_error = None
        try:
            answer_question(vector_store, CASES[0].question, settings)
        except RuntimeError as exc:
            warmup_error = str(exc)
        warmup_seconds = time.perf_counter() - warmup_started

        measured_results = []
        all_latencies: list[float] = []
        for case in CASES:
            latencies: list[float] = []
            outcomes: list[bool] = []
            source_pages: list[int] = []
            failures: list[str] = []
            for _ in range(args.repeats):
                request_started = time.perf_counter()
                try:
                    result = answer_question(vector_store, case.question, settings)
                    pages = [page for _, page in result.sources]
                    outcomes.append(_matches_case(result.answer, pages, case))
                    source_pages.extend(pages)
                except RuntimeError as exc:
                    failures.append(str(exc))
                elapsed = time.perf_counter() - request_started
                latencies.append(elapsed)
                all_latencies.append(elapsed)

            measured_results.append(
                {
                    "case": case,
                    "latencies": latencies,
                    "passed": bool(outcomes) and all(outcomes) and not failures,
                    "source_pages": sorted(set(source_pages)),
                    "failures": failures,
                }
            )

    build_kwh = args.power_watts * build_seconds / 3_600_000
    build_cost = build_kwh * args.electricity_rate_inr_per_kwh
    avg_query_seconds = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    query_cost = (
        args.power_watts
        * avg_query_seconds
        / 3_600_000
        * args.electricity_rate_inr_per_kwh
    )
    amortized_cost = build_cost / args.amortize_index_over_queries
    local_cost_per_query = query_cost + amortized_cost
    total_requests = len(CASES) * args.repeats
    cloud_cost = None
    if args.cloud_cost_per_1000_requests_inr is not None:
        cloud_cost = args.cloud_cost_per_1000_requests_inr * total_requests / 1000

    case_rows = []
    for item in measured_results:
        case = item["case"]
        latencies = item["latencies"]
        mean_latency = sum(latencies) / len(latencies) if latencies else 0
        status = "PASS" if item["passed"] else "FAIL"
        source_text = ", ".join(map(str, item["source_pages"])) or "none"
        failure_text = "; ".join(item["failures"]).replace("|", "\\|") or "-"
        case_rows.append(
            f"| {case.name} | {status} | {source_text} | "
            f"{_format_seconds(mean_latency)} | {failure_text} |"
        )

    pass_count = sum(item["passed"] for item in measured_results)
    latency_sorted = sorted(all_latencies)
    median_latency = statistics.median(latency_sorted) if latency_sorted else 0
    percentile_95 = (
        latency_sorted[min(len(latency_sorted) - 1, math.ceil(len(latency_sorted) * 0.95) - 1)]
        if latency_sorted
        else 0
    )
    cloud_summary = (
        f"Configured cloud baseline: INR {args.cloud_cost_per_1000_requests_inr:g} per 1,000 requests; "
        f"estimated cloud spend for this run: INR {cloud_cost:.6f}; "
        f"estimated run electricity cost: INR {build_cost + query_cost:.6f}."
        if cloud_cost is not None
        else "No hosted API rate was provided; a cloud-cost comparison is intentionally not estimated."
    )
    warmup_summary = (
        f"Warm-up query: {_format_seconds(warmup_seconds)} seconds."
        if warmup_error is None
        else f"Warm-up query failed after {_format_seconds(warmup_seconds)} seconds: {warmup_error}"
    )

    return f"""# RAG POC Evaluation Report

## Run summary

- **Run date (UTC):** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}
- **Platform:** {platform.platform()}
- **Python:** {sys.version.split()[0]}
- **LLM:** Ollama `{settings.ollama_model}` at `{settings.ollama_base_url}`
- **Embedding model:** `{settings.embedding_model}`
- **Source documents/pages/chunks:** {len({str(doc.metadata.get("source", "")) for doc in documents})} / {len(documents)} / {len(chunks)}
- **FAISS index size:** {index_bytes / (1024 * 1024):.2f} MiB (temporary evaluation index)
- **Question repetitions:** {args.repeats}
- **Functional checks:** {pass_count}/{len(CASES)} passed

## Test results

Checks require expected key terms or the exact unknown-answer fallback, plus the expected cited page where specified. These are sample-document regression checks, not a general measure of answer quality.

| Test | Result | Cited page(s) | Mean latency (s) | Error |
|---|---|---:|---:|---|
{chr(10).join(case_rows)}

## Timing

| Measurement | Seconds |
|---|---:|
| PDF loading and chunking | included in index build |
| FAISS build (PDF load, chunking, model initialization/embedding, save) | {_format_seconds(build_seconds)} |
| Saved FAISS index load | {_format_seconds(load_seconds)} |
| First warm-up question | {_format_seconds(warmup_seconds)} |
| Mean question latency (warm-up excluded) | {_format_seconds(avg_query_seconds)} |
| Median question latency | {_format_seconds(median_latency)} |
| Approx. p95 question latency | {_format_seconds(percentile_95)} |

{warmup_summary}

## Cost effectiveness

- **Hosted API charges:** INR 0 for this local Ollama run. The embedding model and LLM execute locally; no hosted inference API is called.
- **Estimated index-build electricity:** INR {build_cost:.6f} ({build_kwh:.8f} kWh).
- **Estimated electricity per question:** INR {query_cost:.8f}, using {args.power_watts:g} W and INR {args.electricity_rate_inr_per_kwh:g}/kWh.
- **Estimated total per question with index-build amortized over {args.amortize_index_over_queries} requests:** INR {local_cost_per_query:.8f}.
- **Comparison:** {cloud_summary}

Electricity is an estimate based on user-supplied device power and energy rate, not a wall-power measurement. Results exclude hardware purchase/depreciation, cooling, idle time, and one-time model downloads. The local approach avoids per-request hosted API fees but shifts costs to local compute and model storage.

## Reproduce

```powershell
python -m scripts.evaluate_poc
```

Tune assumptions, set an optional cloud baseline, or choose another report path:

```powershell
python -m scripts.evaluate_poc --repeats 3 --power-watts 100 --electricity-rate-inr-per-kwh 8 --cloud-cost-per-1000-requests-inr 50 --output reports/evaluation_report.md
```
"""


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--power-watts", type=_nonnegative_float, default=100)
    parser.add_argument(
        "--electricity-rate-inr-per-kwh", type=_nonnegative_float, default=8
    )
    parser.add_argument(
        "--amortize-index-over-queries", type=_positive_int, default=1000
    )
    parser.add_argument(
        "--cloud-cost-per-1000-requests-inr", type=_nonnegative_float
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_evaluation(args)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    _write_report(output, report)
    print(f"Evaluation report written to {output}")
    if "**Live evaluation not run:**" in report:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
