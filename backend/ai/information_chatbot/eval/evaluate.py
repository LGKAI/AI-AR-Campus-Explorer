"""
STELLAR-RAG Evaluation Runner — Dual LLM Mode

Đánh giá RIÊNG BIỆT cho Ollama và Cloud LLM (Groq/DeepSeek/...).
Cùng 1 retrieval context, 2 generator, 2 bộ metric → so sánh trực tiếp.

Cách dùng
    # Chế độ đơn (mặc định, dùng LLM_BACKEND hiện tại)
    python eval/evaluate.py

    # Chế độ kép — đánh giá cả Ollama + Cloud
    python eval/evaluate.py --dual

    # Không LLM judge (nhanh)
    python eval/evaluate.py --dual --no-llm-judge

    # Giới hạn số câu
    python eval/evaluate.py --dual --limit 10

    # Chỉ một category
    python eval/evaluate.py --dual --category medium

    # Tiếp tục từ checkpoint
    python eval/evaluate.py --dual --resume eval/results/eval_XXXXXX.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from metrics import compute_all, aggregate   # noqa: E402

# Terminal colours
BAR = "=" * 70
SEP = "-" * 70

COLORS = {
    "green":  "\033[92m", "yellow": "\033[93m", "red":   "\033[91m",
    "cyan":   "\033[96m", "bold":   "\033[1m",  "reset": "\033[0m",
    "blue":   "\033[94m", "magenta":"\033[95m",
}

def _c(text: str, color: str) -> str:
    return f"{COLORS.get(color,'')}{text}{COLORS['reset']}"

def _score_color(score: float) -> str:
    if score >= 7.5: return "green"
    if score >= 5.0: return "yellow"
    return "red"

def _bar(value: float, width: int = 25) -> str:
    filled = int(round(value / 10 * width))
    return "█" * filled + "░" * (width - filled)

# Progress printer

def _print_progress_dual(
    i: int, total: int, item: dict,
    ollama_score: float, cloud_score: float,
    elapsed: float, cloud_label: str,
) -> None:
    eta    = (elapsed / i) * (total - i) if i > 0 else 0
    winner = ("Ollama" if ollama_score > cloud_score + 0.5
              else cloud_label if cloud_score > ollama_score + 0.5
              else "TIE")
    o_col  = _score_color(ollama_score)
    c_col  = _score_color(cloud_score)
    print(
        f"  [{i:3d}/{total}] {_c(item['id'],'cyan'):8s} "
        f"Ollama={_c(f'{ollama_score:.1f}',o_col):12s} "
        f"{cloud_label}={_c(f'{cloud_score:.1f}',c_col):12s} "
        f"win={_c(winner,'bold'):10s} ETA {eta:.0f}s"
    )

def _print_progress_single(
    i: int, total: int, item: dict, score: float, elapsed: float,
) -> None:
    eta   = (elapsed / i) * (total - i) if i > 0 else 0
    color = _score_color(score)
    print(
        f"  [{i:3d}/{total}] {_c(item['id'],'cyan'):8s} "
        f"score={_c(f'{score:.1f}',color):12s} "
        f"kw={item.get('keyword_coverage',0):.0%} "
        f"ETA {eta:.0f}s"
    )

# Report printers

def _print_report_dual(
    results: list[dict],
    summary_ollama: dict,
    summary_cloud:  dict,
    cloud_label:    str,
    elapsed:        float,
) -> None:
    """Side-by-side comparison report for dual mode."""
    ov_o = summary_ollama["overall"]
    ov_c = summary_cloud["overall"]

    print(f"\n{_c(BAR,'bold')}")
    print(_c(f"  STELLAR-RAG — Dual Evaluation Report", "bold"))
    print(_c(BAR, "bold"))
    print(f"  Thời gian   : {elapsed:.1f}s")
    print(f"  Số câu      : {summary_ollama['total_evaluated']}")
    print(f"  Backends    : Ollama  vs  {cloud_label}")

    metrics_display = [
        ("LLM Overall",      "llm_overall"),
        ("LLM Accuracy",     "llm_accuracy"),
        ("LLM Completeness", "llm_completeness"),
        ("LLM Relevance",    "llm_relevance"),
        ("Composite Score",  "composite_score"),
        ("ROUGE-L F1 ×10",  "rouge_l_f1"),
        ("Keyword Cov ×10", "keyword_coverage"),
    ]

    print(f"\n{_c('  OVERALL SCORES COMPARISON', 'bold')}")
    print(SEP)
    hdr = f"  {'Metric':<22s}  {'Ollama':>8}  {cloud_label:>10}  {'Winner':>8}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    wins = {"ollama": 0, "cloud": 0, "tie": 0}
    for label, key in metrics_display:
        scale = 10 if "f1" in key or "coverage" in key or "similarity" in key else 1
        o_val = ov_o.get(key, {}).get("mean", 0) * (10 if scale == 10 else 1)
        c_val = ov_c.get(key, {}).get("mean", 0) * (10 if scale == 10 else 1)
        diff  = o_val - c_val
        if diff > 0.3:
            winner = "Ollama ✓"
            wins["ollama"] += 1
            w_col = "green"
        elif diff < -0.3:
            winner = f"{cloud_label} ✓"
            wins["cloud"] += 1
            w_col = "blue"
        else:
            winner = "TIE"
            wins["tie"] += 1
            w_col = "yellow"
        o_col = _score_color(o_val)
        c_col = _score_color(c_val)
        print(f"  {label:<22s}  "
              f"{_c(f'{o_val:>6.2f}', o_col):>18s}  "
              f"{_c(f'{c_val:>8.2f}', c_col):>20s}  "
              f"{_c(winner, w_col):>16s}")

    print(f"\n  {'Pass rate ≥6/10':<22s}  "
          f"{ov_o.get('pass_rate_6',0):>8.1%}  "
          f"{ov_c.get('pass_rate_6',0):>10.1%}")
    print(f"  {'Pass rate ≥7/10':<22s}  "
          f"{ov_o.get('pass_rate_7',0):>8.1%}  "
          f"{ov_c.get('pass_rate_7',0):>10.1%}")
    print(f"  {'Citation rate':<22s}  "
          f"{ov_o.get('citation_rate',0):>8.1%}  "
          f"{ov_c.get('citation_rate',0):>10.1%}")

    w_o = wins["ollama"]; w_c = wins["cloud"]; w_t = wins["tie"]
    print(f"\n  {'WINS':<22s}  "
          f"{_c(f'Ollama: {w_o}', 'green')}  "
          f"{_c(f'{cloud_label}: {w_c}', 'blue')}  "
          f"{_c(f'Tie: {w_t}', 'yellow')}")

    # Per-category
    cat_order  = ["medium", "hard", "comprehensive"]
    cat_labels = {"medium": "Medium", "hard": "Hard", "comprehensive": "Comprehensive"}

    print(f"\n{_c('  BY CATEGORY', 'bold')}")
    print(SEP)
    for cat in cat_order:
        co = summary_ollama["by_category"].get(cat, {})
        cc = summary_cloud["by_category"].get(cat, {})
        if not co and not cc:
            continue
        o_llm = co.get("llm_overall", {}).get("mean", 0)
        c_llm = cc.get("llm_overall", {}).get("mean", 0)
        o_col = _score_color(o_llm)
        c_col = _score_color(c_llm)
        cnt   = co.get("count", cc.get("count", 0))
        print(f"\n  {_c(cat_labels[cat], 'bold')} (n={cnt})")
        print(f"    Ollama LLM Overall : {_c(f'{o_llm:.2f}/10', o_col)}  {_bar(o_llm, 20)}")
        print(f"    {cloud_label} LLM Overall : {_c(f'{c_llm:.2f}/10', c_col)}  {_bar(c_llm, 20)}")

    # Per-question detail table
    print(f"\n{_c('  PER-QUESTION DETAIL', 'bold')}")
    print(SEP)
    print(f"  {'ID':8s}  {'Ollama':>7}  {cloud_label:>9}  {'Diff':>6}  Question")
    print("  " + "─" * 66)
    for r in results:
        o  = r.get("ollama_llm_overall", 0)
        c  = r.get("cloud_llm_overall",  0)
        d  = o - c
        d_str  = f"{d:+.1f}"
        d_col  = "green" if d > 0.3 else ("blue" if d < -0.3 else "yellow")
        print(f"  {r['id']:8s}  {_c(f'{o:.1f}',_score_color(o)):>17s}  "
              f"{_c(f'{c:.1f}',_score_color(c)):>19s}  "
              f"{_c(d_str,d_col):>14s}  {r['question'][:38]}")

    print(f"\n{_c(BAR,'bold')}\n")

def _print_report_single(results: list[dict], summary: dict, elapsed: float) -> None:
    """Standard single-LLM report."""
    cat_order  = ["medium", "hard", "comprehensive"]
    cat_labels = {"medium": "Medium", "hard": "Hard", "comprehensive": "Comprehensive"}

    print(f"\n{_c(BAR,'bold')}")
    print(_c("  STELLAR-RAG — Evaluation Report", "bold"))
    print(_c(BAR,'bold'))
    print(f"  Thời gian : {elapsed:.1f}s  |  {summary['total_evaluated']} câu")

    ov = summary["overall"]
    metrics_display = [
        ("LLM Overall",      "llm_overall",      1),
        ("LLM Accuracy",     "llm_accuracy",     1),
        ("LLM Completeness", "llm_completeness", 1),
        ("Composite Score",  "composite_score",  1),
        ("ROUGE-L F1",       "rouge_l_f1",       10),
        ("Keyword Coverage", "keyword_coverage", 10),
    ]
    print(f"\n{_c('  OVERALL', 'bold')}")
    print(SEP)
    for label, key, scale in metrics_display:
        if key not in ov:
            continue
        v     = ov[key]["mean"] * (10 if scale == 10 else 1)
        color = _score_color(v)
        print(f"  {label:<22s} {_c(f'{v:5.2f}', color)}/10  {_bar(v)}")

    print(f"\n  Pass ≥6/10 : {ov.get('pass_rate_6',0):.1%}")
    print(f"  Pass ≥7/10 : {ov.get('pass_rate_7',0):.1%}")

    print(f"\n{_c('  BY CATEGORY', 'bold')}")
    print(SEP)
    for cat in cat_order:
        if cat not in summary["by_category"]:
            continue
        c = summary["by_category"][cat]
        llm  = c.get("llm_overall", {}).get("mean", 0)
        comp = c.get("composite_score", {}).get("mean", 0)
        print(f"\n  {_c(cat_labels[cat],'bold')} (n={c['count']})")
        print(f"    LLM Overall  : {_c(f'{llm:.2f}/10', _score_color(llm))}  {_bar(llm, 20)}")
        print(f"    Composite    : {comp:.2f}/10")

    print(f"\n{_c(BAR,'bold')}\n")

# ─
# Core evaluation loops
# ─

def run_evaluation_dual(
    questions:      list[dict],
    use_llm_judge:  bool = True,
    resume_results: list[dict] | None = None,
    save_path:      Path | None = None,
) -> tuple[list[dict], dict, dict, str]:
    """
    Dual-mode: call answer_dual() for each question.
    Returns (results, summary_ollama, summary_cloud, cloud_label).
    """
    import ollama as ollama_lib

    print(f"\n{_c(BAR,'bold')}")
    print(_c("  STELLAR-RAG — Dual Evaluation (Ollama + Cloud)", "bold"))
    print(_c(BAR,'bold'))

    t0 = time.time()
    from agent  import Agent
    from config import settings
    agent = Agent()
    # Ensure both backends ready
    agent.llm_client.ensure_dual()

    ollama_client = ollama_lib.Client() if use_llm_judge else None
    judge_model   = settings.ollama_model
    cloud_label   = settings.cloud_provider.capitalize()
    cloud_model   = getattr(agent.llm_client._cloud, "model", "?") if agent.llm_client._cloud else "N/A"

    print(f"  Agent loaded in {time.time()-t0:.1f}s")
    print(f"  Ollama : {settings.ollama_model}")
    print(f"  Cloud  : {cloud_label} / {cloud_model}\n")

    done_ids: set[str] = set()
    results: list[dict] = []
    if resume_results:
        results   = list(resume_results)
        done_ids  = {r["id"] for r in results}
        print(f"  Resuming: {len(done_ids)} done, {len(questions)-len(done_ids)} remaining\n")

    pending    = [q for q in questions if q["id"] not in done_ids]
    total      = len(pending)
    eval_start = time.time()

    print(f"  {_c(f'Evaluating {total} questions — DUAL MODE', 'bold')}")
    print(SEP)

    for i, item in enumerate(pending, 1):
        qid  = item["id"]
        q    = item["question"]
        t_q  = time.time()   # per-question timer (fixes operator-precedence bug)

        # Run dual agent
        try:
            ollama_ans, cloud_ans, _tid = agent.answer_dual(q)
        except Exception as exc:
            ollama_ans = f"[OLLAMA ERROR: {exc}]"
            cloud_ans  = f"[CLOUD ERROR: {exc}]"

        # Metrics for Ollama
        m_ollama = compute_all(
            item          = item,
            agent_answer  = ollama_ans,
            ollama_client = ollama_client,
            model         = judge_model,
            use_llm_judge = use_llm_judge,
        )

        # Metrics for Cloud
        m_cloud = compute_all(
            item          = item,
            agent_answer  = cloud_ans,
            ollama_client = ollama_client,
            model         = judge_model,
            use_llm_judge = use_llm_judge,
        )

        # Merge into single result row
        row: dict = {
            "id":          qid,
            "category":    item["category"],
            "question":    q,
            "gold_answer": item["gold_answer"],
            "eval_time_s": round(time.time() - t_q, 2),   # wall-clock for this question only
        }
        for k, v in m_ollama.items():
            if k not in ("id", "category", "question", "gold_answer"):
                row[f"ollama_{k}"] = v
        for k, v in m_cloud.items():
            if k not in ("id", "category", "question", "gold_answer"):
                row[f"cloud_{k}"] = v

        # Convenience aliases for report
        row["ollama_llm_overall"] = m_ollama.get("llm_overall", 0)
        row["cloud_llm_overall"]  = m_cloud.get("llm_overall",  0)

        results.append(row)

        # Progress
        elapsed = time.time() - eval_start
        _print_progress_dual(i, total, row,
                             row["ollama_llm_overall"],
                             row["cloud_llm_overall"],
                             elapsed, cloud_label)

        if save_path and i % 5 == 0:
            _autosave(results, save_path)

    # Build separate summaries
    # Reformat as standard metric rows for aggregate()
    ollama_rows = _extract_rows(results, prefix="ollama_")
    cloud_rows  = _extract_rows(results, prefix="cloud_")

    summary_ollama = aggregate(ollama_rows)
    summary_cloud  = aggregate(cloud_rows)

    for s in (summary_ollama, summary_cloud):
        s["eval_config"] = {
            "use_llm_judge": use_llm_judge,
            "dual_mode":     True,
            "ollama_model":  judge_model,
            "cloud_label":   cloud_label,
            "cloud_model":   cloud_model,
            "timestamp":     datetime.now().isoformat(),
        }

    return results, summary_ollama, summary_cloud, cloud_label

def _extract_rows(results: list[dict], prefix: str) -> list[dict]:
    """Re-map prefixed keys back to standard metric keys for aggregate()."""
    rows = []
    for r in results:
        row = {
            "id":          r["id"],
            "category":    r["category"],
            "question":    r["question"],
            "gold_answer": r.get("gold_answer", ""),
        }
        for k, v in r.items():
            if k.startswith(prefix):
                row[k[len(prefix):]] = v
        rows.append(row)
    return rows

def run_evaluation_single(
    questions:      list[dict],
    use_llm_judge:  bool = True,
    resume_results: list[dict] | None = None,
    save_path:      Path | None = None,
) -> tuple[list[dict], dict]:
    """Single-LLM mode (original behaviour)."""
    import ollama as ollama_lib

    print(f"\n{_c(BAR,'bold')}")
    print(_c("  STELLAR-RAG — Single LLM Evaluation", "bold"))
    print(_c(BAR,'bold'))

    t0 = time.time()
    from agent  import Agent
    from config import settings
    agent         = Agent()
    ollama_client = ollama_lib.Client() if use_llm_judge else None
    model         = settings.ollama_model
    print(f"  Agent loaded in {time.time()-t0:.1f}s  |  LLM: {model}\n")

    done_ids: set[str] = set()
    results: list[dict] = []
    if resume_results:
        results  = list(resume_results)
        done_ids = {r["id"] for r in results}

    pending    = [q for q in questions if q["id"] not in done_ids]
    total      = len(pending)
    eval_start = time.time()

    print(f"  {_c(f'Evaluating {total} questions', 'bold')}  (LLM judge: {'ON' if use_llm_judge else 'OFF'})")
    print(SEP)

    for i, item in enumerate(pending, 1):
        t_q = time.time()   # per-question timer (fixes operator-precedence bug)
        try:
            ans, _ = agent.answer(item["question"])
        except Exception as exc:
            ans = f"[ERROR: {exc}]"

        metrics = compute_all(
            item=item, agent_answer=ans,
            ollama_client=ollama_client, model=model,
            use_llm_judge=use_llm_judge,
        )
        metrics["eval_time_s"] = round(time.time() - t_q, 2)   # wall-clock for this question only
        results.append(metrics)

        elapsed = time.time() - eval_start
        _print_progress_single(i, total, metrics, metrics.get("llm_overall", 0), elapsed)

        if save_path and i % 5 == 0:
            _autosave(results, save_path)

    summary = aggregate(results)
    summary["eval_config"] = {
        "use_llm_judge": use_llm_judge,
        "dual_mode":     False,
        "model":         model,
        "timestamp":     datetime.now().isoformat(),
    }
    return results, summary

# ─
# Save helpers
# ─

def _autosave(results: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def save_results(
    results: list[dict],
    summary: dict | tuple[dict, dict],
    out_dir: Path,
    dual: bool = False,
    cloud_label: str = "Cloud",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "dual" if dual else "single"
    path = out_dir / f"eval_{mode}_{ts}.json"

    payload: dict = {"results": results}
    if dual and isinstance(summary, tuple):
        payload["summary_ollama"] = summary[0]
        payload["summary_cloud"]  = summary[1]
        payload["cloud_label"]    = cloud_label
    else:
        payload["summary"] = summary if not isinstance(summary, tuple) else summary[0]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  Results saved → {path}")
    return path

def _export_csv(results: list[dict], path: Path, dual: bool = False) -> None:
    if dual:
        cols = [
            "id", "category",
            "ollama_llm_overall", "ollama_llm_accuracy", "ollama_composite_score",
            "ollama_rouge_l_f1", "ollama_keyword_coverage",
            "cloud_llm_overall",  "cloud_llm_accuracy",  "cloud_composite_score",
            "cloud_rouge_l_f1",  "cloud_keyword_coverage",
            "eval_time_s",
        ]
    else:
        cols = [
            "id", "category", "llm_overall", "llm_accuracy", "llm_completeness",
            "composite_score", "rouge_l_f1", "keyword_coverage",
            "citation_ok", "exact_match", "eval_time_s",
        ]
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="STELLAR-RAG Evaluation")
    parser.add_argument("--dataset",      default="eval/qa_dataset.json")
    parser.add_argument("--category",     choices=["medium","hard","comprehensive","all"], default="all")
    parser.add_argument("--limit",        type=int, default=0,
                        help="Stop after N questions (0 = all)")
    parser.add_argument("--start",        type=int, default=1,
                        help="Start from question number N (1-based). "
                             "Use with --resume to append to existing results.")
    parser.add_argument("--dual",         action="store_true",  help="Evaluate Ollama AND Cloud LLM side-by-side")
    parser.add_argument("--no-llm-judge", action="store_true",  help="Skip LLM judge (faster)")
    parser.add_argument("--resume",       default="",
                        help="Path to a prior result JSON to append to (use with --start)")
    parser.add_argument("--out-dir",      default="eval/results")
    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}"); sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]
    if args.category != "all":
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit > 0:
        questions = questions[:args.limit]

    # --start: skip to question index N (1-based)
    start_idx = max(1, args.start) - 1   # convert to 0-based
    if start_idx > 0:
        if start_idx >= len(questions):
            print(f"[ERROR] --start {args.start} exceeds dataset size ({len(questions)})")
            sys.exit(1)
        questions = questions[start_idx:]

    total_in_set = start_idx + len(questions)
    print(f"\n  Dataset: {dataset_path.name}  |  {total_in_set} questions total")
    if start_idx > 0:
        print(f"  Start  : Q{args.start} (skipping first {start_idx})")
    print(f"  Running: {len(questions)} questions  ({start_idx+1}–{total_in_set})")
    print(f"  Mode   : {'DUAL (Ollama + Cloud)' if args.dual else 'SINGLE'}")
    print(f"  Judge  : {'OFF (no-llm-judge)' if args.no_llm_judge else 'ON (Ollama LLM judge)'}")

    # Resume / append: load prior results to merge into final output
    resume_results: list[dict] = []
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.out_dir) / f"eval_{'dual' if args.dual else 'single'}_{ts}.json"

    if args.resume:
        rpath = Path(args.resume)
        if rpath.exists():
            with open(rpath, encoding="utf-8") as f:
                old = json.load(f)
            resume_results = old.get("results", [])
            save_path = rpath   # write back to the same file
            done_ids  = {r["id"] for r in resume_results}
            print(f"  Resume : {rpath.name}  ({len(resume_results)} prior results loaded)")
            # Filter out any IDs that were already evaluated
            questions = [q for q in questions if q["id"] not in done_ids]
            print(f"  After dedup: {len(questions)} questions remaining")

    # Run
    t_start = time.time()

    if args.dual:
        results, sum_ollama, sum_cloud, cloud_label = run_evaluation_dual(
            questions      = questions,
            use_llm_judge  = not args.no_llm_judge,
            resume_results = resume_results,
            save_path      = save_path,
        )
        elapsed = time.time() - t_start
        _print_report_dual(results, sum_ollama, sum_cloud, cloud_label, elapsed)
        final_path = save_results(results, (sum_ollama, sum_cloud),
                                  Path(args.out_dir), dual=True, cloud_label=cloud_label)
    else:
        results, summary = run_evaluation_single(
            questions      = questions,
            use_llm_judge  = not args.no_llm_judge,
            resume_results = resume_results,
            save_path      = save_path,
        )
        elapsed = time.time() - t_start
        _print_report_single(results, summary, elapsed)
        final_path = save_results(results, summary, Path(args.out_dir), dual=False)

    # CSV
    csv_path = final_path.with_suffix(".csv")
    _export_csv(results, csv_path, dual=args.dual)
    print(f"  CSV saved → {csv_path}\n")

if __name__ == "__main__":
    main()
