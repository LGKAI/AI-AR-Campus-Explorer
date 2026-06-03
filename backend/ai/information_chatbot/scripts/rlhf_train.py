#!/usr/bin/env python3
"""
rlhf_train.py — Offline RLHF training script for STELLAR-RAG v4.

Reads all user feedback stored in storage/memory.db, performs a batch
QDAP-S update, and exports fine-tuning data as JSONL files.

Run this script periodically (e.g. nightly via cron/task scheduler) to
bake accumulated user feedback into the retrieval model.

Usage
-----
    # Full batch update + JSONL export
    python rlhf_train.py

    # Process only NEW feedback since the last run (faster)
    python rlhf_train.py --incremental

    # Export JSONL only, skip QDAP-S batch update
    python rlhf_train.py --export-only

    # Show statistics without writing anything
    python rlhf_train.py --dry-run

    # Require at least N samples before running (default: 5)
    python rlhf_train.py --min-samples 10

    # Change the JSONL export directory
    python rlhf_train.py --export-dir eval/rlhf_exports

Output
------
    storage/rlhf_state.json                  — checkpoint (last processed row id)
    eval/rlhf_exports/preference_pairs.jsonl — DPO pairs (chosen / rejected)
    eval/rlhf_exports/sft_positives.jsonl    — SFT pairs (reward >= 4)

External fine-tuning (using the exported JSONL files):
    Axolotl  : axolotl train config.yml  (dataset type 'preference')
    TRL / DPO: python -m trl dpo --model <model> --dataset preference_pairs.jsonl
    LLaMA-F  : llamafactory-cli train --data sft_positives.jsonl

Notes
-----
    - This script does NOT fine-tune the LLM directly (Ollama models
      cannot be fine-tuned in-place).
    - Online RLHF (1-5 rating after each answer) already runs in real-time
      inside app.py.
    - This script focuses on: (1) batch QDAP-S re-learning, (2) dataset export.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add src/ to Python path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from config import settings
from rlhf import (
    RLHFReport,
    build_preference_pairs,
    batch_qdap_update,
    compute_reward_weights,
    export_jsonl,
    load_feedback,
    load_feedback_since,
    load_rlhf_state,
    save_rlhf_state,
)

# Helpers

def _banner():
    print("=" * 62)
    print("  STELLAR-RAG v4 — RLHF Offline Training")
    print("=" * 62)

def _load_graphrag_and_embedder():
    """Load GraphRAG + Embedder (required only for QDAP batch update)."""
    from graphrag  import GraphRAG
    from embedding import Embedder

    print("[RLHF] Loading GraphRAG…")
    gr = GraphRAG()
    if not gr.exists():
        print(
            "[RLHF] ERROR: FAISS index not found.\n"
            "       Run: python ingest.py"
        )
        sys.exit(1)
    gr.load()
    embedder = Embedder()
    return gr, embedder

# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="STELLAR-RAG RLHF — batch update from user feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Process only new feedback since the last run",
    )
    parser.add_argument(
        "--export-only", action="store_true",
        help="Export JSONL only, skip QDAP-S batch update",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show statistics without writing anything",
    )
    parser.add_argument(
        "--min-samples", type=int, default=5, metavar="N",
        help="Minimum number of samples required to run (default: 5)",
    )
    parser.add_argument(
        "--export-dir", type=str, default="eval/rlhf_exports",
        help="JSONL export directory (default: eval/rlhf_exports)",
    )
    args = parser.parse_args()

    _banner()

    db_path    = settings.memory_db_path
    state      = load_rlhf_state()
    last_id    = state.get("last_processed_id", 0) if args.incremental else 0
    export_dir = Path(args.export_dir)
    report     = RLHFReport()

    # 1. Load feedback
    if args.incremental:
        print(f"[RLHF] Mode: incremental (id > {last_id})")
        rows = load_feedback_since(db_path, since_id=last_id)
    else:
        print("[RLHF] Mode: full batch")
        rows = load_feedback(db_path)

    report.total_feedback = len(rows)
    report.positive_count = sum(1 for r in rows if r.reward >= 4)
    report.negative_count = sum(1 for r in rows if r.reward <= 2)
    report.neutral_count  = sum(1 for r in rows if r.reward == 3)

    print(f"\n[RLHF] Feedback loaded : {report.total_feedback} rows")
    print(f"       Positive (4-5)  : {report.positive_count}")
    print(f"       Negative (1-2)  : {report.negative_count}")
    print(f"       Neutral    (3)  : {report.neutral_count}")

    if report.total_feedback < args.min_samples:
        print(
            f"\n[RLHF] WARNING:  Not enough samples "
            f"({report.total_feedback} < {args.min_samples}).\n"
            "       Collect more feedback in app.py, then re-run this script."
        )
        return

    # 2. Build preference pairs
    pairs = build_preference_pairs(rows)
    report.pairs_built = len(pairs)
    print(f"\n[RLHF] DPO preference pairs: {report.pairs_built}")

    # 3. Keyword reward weights
    weights = compute_reward_weights(rows, top_n=50)
    if weights:
        top5 = list(weights.items())[:5]
        print(f"[RLHF] Top keyword weights: {top5}")

    if args.dry_run:
        print("\n[RLHF] Dry-run complete — nothing written.")
        return

    # 4. Export JSONL
    if rows:
        exported_dir = export_jsonl(pairs, rows, export_dir)
        report.exported_path = str(exported_dir)

    # 5. Batch QDAP-S update
    if not args.export_only:
        graphrag, embedder = _load_graphrag_and_embedder()
        n_updates = batch_qdap_update(rows, graphrag, embedder)
        report.qdap_updates_run = n_updates
        print(f"\n[RLHF] QDAP-S updates applied: {n_updates}")

        # Persist QDAP weights if supported
        try:
            graphrag.save_qdap_weights()
            print("[RLHF] QDAP weights saved to disk.")
        except AttributeError:
            print("[RLHF] WARNING:  save_qdap_weights() not implemented — skipping persist.")
            print("       (QDAP weights will reset to default on next app restart)")
    else:
        print("[RLHF] export-only mode: skipping QDAP batch update.")

    # 6. Save checkpoint
    last_processed_id = rows[-1].id if rows else last_id
    save_rlhf_state(last_processed_id, report)

    # 7. Summary
    print("\n" + "=" * 62)
    print("  RLHF Training Summary")
    print("=" * 62)
    print(f"  Feedback processed : {report.total_feedback}")
    print(f"  DPO pairs exported : {report.pairs_built}")
    print(f"  QDAP updates       : {report.qdap_updates_run}")
    print(f"  Export directory   : {report.exported_path}")
    print(f"  Checkpoint         : storage/rlhf_state.json")
    print("=" * 62)
    print("\n[RLHF] Done.")

if __name__ == "__main__":
    main()
