"""
RLHF utilities — STELLAR-RAG v4.

Provides:
  • load_feedback()          — read all feedback from SQLite
  • load_feedback_since()    — read incrementally (only new rows)
  • build_preference_pairs() — (chosen, rejected) pairs for DPO
  • batch_qdap_update()      — batch replay through the QDAP-S predictor
  • compute_reward_weights() — keyword weights from historical feedback
  • export_jsonl()           — export JSONL for external fine-tuning
  • save/load_rlhf_state()   — checkpoint of the last processed row id

Usage:
  Online  → app.py calls update_qdap_online() after each rating (built-in).
  Offline → rlhf_train.py calls batch_qdap_update() to replay full history.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# ─
# Data models
# ─

@dataclass
class FeedbackRow:
    id:               int
    ts:               str
    turn_id:          str
    user_query:       str
    assistant_answer: str
    reward:           int   # 1–5 (user rating)
    note:             str

@dataclass
class PreferencePair:
    """DPO-style pair: good answer vs. poor answer for the same question."""
    query:            str
    chosen:           str   # reward >= 4
    rejected:         str   # reward <= 2
    chosen_reward:    int
    rejected_reward:  int

@dataclass
class RLHFReport:
    total_feedback:   int = 0
    positive_count:   int = 0   # rewards 4-5
    negative_count:   int = 0   # rewards 1-2
    neutral_count:    int = 0   # reward 3 (neutral)
    pairs_built:      int = 0
    qdap_updates_run: int = 0
    exported_path:    str = ""
    timestamp:        str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

# ─
# Load feedback from SQLite
# ─

def load_feedback(db_path: Path) -> list[FeedbackRow]:
    """Read the entire feedback table."""
    if not db_path.exists():
        print(f"[RLHF] WARNING:  DB không tồn tại: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute(
        "SELECT id, ts, turn_id, user_query, assistant_answer, reward, "
        "COALESCE(note, '') FROM feedback ORDER BY id"
    )
    rows = [FeedbackRow(*row) for row in cur.fetchall()]
    conn.close()
    return rows

def load_feedback_since(db_path: Path, since_id: int = 0) -> list[FeedbackRow]:
    """Read only feedback rows newer than since_id (incremental)."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute(
        "SELECT id, ts, turn_id, user_query, assistant_answer, reward, "
        "COALESCE(note, '') FROM feedback WHERE id > ? ORDER BY id",
        (since_id,),
    )
    rows = [FeedbackRow(*row) for row in cur.fetchall()]
    conn.close()
    return rows

# ─
# Build preference pairs (DPO)
# ─

def build_preference_pairs(
    rows: list[FeedbackRow],
    positive_threshold: int = 4,
    negative_threshold: int = 2,
) -> list[PreferencePair]:
    """
    Create (chosen, rejected) pairs from the feedback list.

    Algorithm:
      1. Group rows by user_query (lowercased).
      2. Within each group, take the highest-rated answer (chosen) and lowest (rejected).
      3. Only create a pair when the group has both positive and negative examples.

    This produces DPO (Direct Preference Optimization) format.
    """
    groups: dict[str, list[FeedbackRow]] = defaultdict(list)
    for row in rows:
        key = row.user_query.strip().lower()
        groups[key].append(row)

    pairs: list[PreferencePair] = []
    for _key, group_rows in groups.items():
        positives = [r for r in group_rows if r.reward >= positive_threshold]
        negatives = [r for r in group_rows if r.reward <= negative_threshold]
        if not positives or not negatives:
            continue

        best  = max(positives, key=lambda r: r.reward)
        worst = min(negatives, key=lambda r: r.reward)
        pairs.append(PreferencePair(
            query=best.user_query,
            chosen=best.assistant_answer,
            rejected=worst.assistant_answer,
            chosen_reward=best.reward,
            rejected_reward=worst.reward,
        ))

    return pairs

# ─
# Batch QDAP-S update from feedback history
# ─

def batch_qdap_update(
    rows: list[FeedbackRow],
    graphrag,   # GraphRAG instance (passed in to avoid circular import)
    embedder,   # Embedder instance
) -> int:
    """
    Replay the full feedback history through the QDAP-S predictor.

    For each feedback row:
      1. Encode user_query into a vector.
      2. Set graphrag._last_qv = vector (simulates the original query).
      3. Call graphrag.update_qdap_online(reward).

    rating 1-5 → reward [-1, +1]:  (rating - 3) / 2.0
    rating 3   → skipped (neutral)

    Returns the number of updates applied.
    """
    if not hasattr(graphrag, "_qdap_predictor") or graphrag._qdap_predictor is None:
        print("[RLHF] WARNING:  QDAP predictor chưa được khởi tạo — chạy ingest.py trước.")
        return 0

    n_updates = 0
    for row in rows:
        if row.reward == 3:
            continue   # neutral — skip

        try:
            qv = embedder.encode([row.user_query])   # shape (1, d)
            graphrag._last_qv = qv
            reward = (row.reward - 3) / 2.0          # map [1,5] to [-1,+1]
            graphrag.update_qdap_online(reward)
            n_updates += 1
        except Exception as exc:
            print(f"[RLHF] WARNING:  Skipping row {row.id}: {exc}")

    return n_updates

# ─
# Keyword weights (retrieval-level RLHF signal)
# ─

def compute_reward_weights(
    rows: list[FeedbackRow],
    top_n: int = 50,
) -> dict[str, float]:
    """
    Compute reward-weighted keyword scores from historical feedback.

    Keywords in highly-rated queries → weight > 0.
    Keywords in poorly-rated queries → weight < 0.

    Weights are in [-1, +1] and can be used to boost BM25.
    Only keywords with >= 2 samples are kept (to avoid noise).
    """
    keyword_rewards: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        reward = (row.reward - 3) / 2.0
        tokens = re.findall(r"\w{3,}", row.user_query.lower())
        for tok in tokens:
            keyword_rewards[tok].append(reward)

    weights = {
        kw: float(np.mean(rw))
        for kw, rw in keyword_rewards.items()
        if len(rw) >= 2
    }

    # Sort by |weight| descending, keep top_n
    weights = dict(
        sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    )
    return weights

# ─
# Export JSONL for external fine-tuning
# ─

def export_jsonl(
    pairs: list[PreferencePair],
    rows: list[FeedbackRow],
    out_dir: Path,
) -> Path:
    """
    Export 2 JSONL files to out_dir:

    1. preference_pairs.jsonl  — DPO training pairs (chosen / rejected)
    2. sft_positives.jsonl     — SFT, only rows with reward >= 4

    Compatible with Axolotl, LLaMA-Factory, and TRL.
    Returns out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. DPO preference pairs
    dpo_path = out_dir / "preference_pairs.jsonl"
    with dpo_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            record = {
                "prompt":          pair.query,
                "chosen":          pair.chosen,
                "rejected":        pair.rejected,
                "chosen_reward":   pair.chosen_reward,
                "rejected_reward": pair.rejected_reward,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[RLHF]  DPO pairs ({len(pairs)}) → {dpo_path}")

    # 2. SFT positives
    sft_path  = out_dir / "sft_positives.jsonl"
    positives = [r for r in rows if r.reward >= 4]
    with sft_path.open("w", encoding="utf-8") as f:
        for row in positives:
            record = {
                "instruction": row.user_query,
                "output":      row.assistant_answer,
                "reward":      row.reward,
                "note":        row.note,
                "ts":          row.ts,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[RLHF]  SFT positives ({len(positives)}) → {sft_path}")

    return out_dir

# ─
# Checkpoint — save/load the last processed feedback row id
# ─

_RLHF_STATE_FILE = Path("storage") / "rlhf_state.json"

def save_rlhf_state(last_id: int, report: RLHFReport) -> None:
    """Save checkpoint for incremental runs."""
    state = {
        "last_processed_id": last_id,
        "report":            report.__dict__,
    }
    _RLHF_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _RLHF_STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def load_rlhf_state() -> dict:
    """Load checkpoint. Returns empty dict if none exists."""
    if _RLHF_STATE_FILE.exists():
        return json.loads(_RLHF_STATE_FILE.read_text(encoding="utf-8"))
    return {"last_processed_id": 0}
