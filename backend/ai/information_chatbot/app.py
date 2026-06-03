"""
STELLAR-RAG v4 — Interactive chat entry point.

Usage:
    python app.py

Starts the terminal chat loop. Asks the user to choose an input mode:
  T — Text  : type queries normally
  S — Speech: record voice → STT, answer → TTS played through speaker

After each answer, the user can rate 1-5 to reinforce QDAP-S.
"""
from __future__ import annotations

import io
import os
import sys

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent  import Agent
from config import settings

# ─
# Input mode selection
# ─

def _select_mode() -> str:
    """Ask the user to select input mode T (Text) or S (Speech). Returns 'T' or 'S'."""
    print("=" * 50)
    print("  STELLAR-RAG v4 — EHRAG + HybGRAG")
    print("=" * 50)
    print("\nChọn chế độ nhập liệu:")
    print("  T — Text   (gõ phím như bình thường)")
    print("  S — Speech (ghi âm giọng nói tiếng Việt)\n")

    while True:
        choice = input("Nhập T hoặc S: ").strip().upper()
        if choice in {"T", "S"}:
            return choice
        print("  WARNING:  Vui lòng nhập T hoặc S.")

# ─
# Speech initialisation (lazy — only loaded when needed)
# ─

def _init_speech():
    """Check and import the speech module. Returns (listen_fn, speak_fn, unload_fn) or None if dependencies are missing."""
    try:
        from speech import listen, speak, unload_stt, DEFAULT_MODEL
        import torch
        device_lbl = (
            f"GPU ({torch.cuda.get_device_name(0)}, fp16)"
            if torch.cuda.is_available() else "CPU (fp32)"
        )
        print("\n[Speech] Đang kiểm tra module giọng nói…")
        print(f"[Speech] STT: PhoWhisper-medium  [{device_lbl}]")
        print("[Speech] TTS: edge-tts vi-VN-HoaiMyNeural (HTTP)")
        print("[Speech]  Sẵn sàng. Mô hình STT sẽ tải lần đầu dùng.\n")
        return listen, speak, unload_stt
    except ImportError as exc:
        print(f"\n[Speech]  Lỗi import: {exc}")
        print("  Cài đặt: pip install transformers sounddevice soundfile edge-tts pygame")
        return None

# ─
# Main chat loop
# ─

def main() -> None:
    settings.ensure_dirs()
    agent = Agent()

    # Select mode T / S
    mode = _select_mode()

    listen_fn  = None
    speak_fn   = None
    unload_fn  = None   # free VRAM after STT, before Ollama runs

    if mode == "S":
        speech_fns = _init_speech()
        if speech_fns is None:
            print("WARNING:  Chuyển về chế độ Text do thiếu thư viện speech.")
            mode = "T"
        else:
            listen_fn, speak_fn, unload_fn = speech_fns

    # Select answer mode: single (1 LLM) or dual (Ollama + Cloud LLM)
    print("\nChọn chế độ trả lời:")
    print("  1 — Đơn  (1 LLM theo LLM_BACKEND)")
    print("  2 — Kép  (Ollama + Cloud LLM song song)\n")
    dual_mode = False
    while True:
        rc = input("Nhập 1 hoặc 2: ").strip()
        if rc == "1":
            break
        if rc == "2":
            dual_mode = True
            break
        print("  WARNING:  Vui lòng nhập 1 hoặc 2.")

    # Display header
    print("\n" + "=" * 50)
    if mode == "T":
        print("  Chế độ TEXT — gõ 'exit' để thoát.")
    else:
        print("  Chế độ SPEECH — nói rồi nhấn Enter để dừng ghi âm.")
        print("  Gõ 'exit' bất cứ lúc nào để thoát.")
    ans_label = "Ollama + Cloud LLM (kép)" if dual_mode else "1 LLM"
    print(f"  Trả lời: {ans_label}")
    print("  Sau mỗi câu trả lời, rate 1-5 để cải thiện hệ thống.")
    print("  Gõ '?debug' để xem context + chunks gửi vào LLM.")
    print("  Gõ '?clear-memory' để xóa lịch sử hội thoại bị nhiễm.")
    print("=" * 50 + "\n")

    # Conversation loop
    while True:
        try:
            if mode == "T":
                # Text mode
                q = input("You> ").strip()
            else:
                # Speech mode
                # Allow typing exit before recording starts
                print("You> (nhấn Enter để bắt đầu ghi âm, hoặc gõ 'exit')")
                pre = input().strip().lower()
                if pre in {"exit", "quit"}:
                    q = "exit"
                else:
                    q = listen_fn()   # record → STT → text
                    # Free PhoWhisper VRAM before Ollama runs
                    if unload_fn is not None:
                        unload_fn()

        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if not q or q.lower() in {"exit", "quit"}:
            print("Tạm biệt!")
            break

        # Special commands
        if q.strip() == "?clear-memory":
            agent.memory.clear()
            agent.cache.clear()
            print(" Đã xóa toàn bộ memory + cache.\n")
            continue

        if q.strip() == "?debug":
            di = agent.debug_info
            print(f"\n{'═'*60}")
            print(f"[DEBUG] complexity={di.get('query_complexity','?')}  "
                  f"dense={di.get('num_dense_hits',0)}  "
                  f"ctx_len={di.get('context_length',0)}c  "
                  f"critic_iter={di.get('critic_iterations',0)}")
            print(f"\n── Dense hits (score | source | page) ──")
            for i, h in enumerate(di.get("dense_hits", [])[:8], 1):
                print(f"  {i}. score={h.get('score',0):.4f} | "
                      f"{h.get('source','?')} tr.{h.get('page','?')} | "
                      f"{h.get('text','')[:120].replace(chr(10),' ')}")
            print(f"\n── Context gửi vào LLM ──")
            print(di.get("context", "(trống)")[:3000])
            print(f"\n── Prompt user message ──")
            msgs = di.get("llm_messages", [])
            for m in msgs:
                if m.get("role") == "user":
                    print(m["content"][:2000])
            print(f"{'═'*60}\n")
            continue

        # Call Agent
        if dual_mode:
            ollama_ans, cloud_ans, turn_id = agent.answer_dual(q)
            cloud_label = settings.cloud_provider.capitalize()
            cloud_model = settings.cloud_model or "(provider default)"
            print(f"\n{'─'*50}")
            print(f"[Ollama — {settings.ollama_model}]")
            print(f"{ollama_ans}")
            print(f"\n{'─'*50}")
            print(f"[{cloud_label} — {cloud_model}]")
            print(f"{cloud_ans}")
            print(f"{'─'*50}\n")
            ans = ollama_ans   # used for TTS + rating
        else:
            ans, turn_id = agent.answer(q)
            print(f"\nAgent> {ans}\n")

        # TTS output (Speech mode)
        if mode == "S" and speak_fn is not None:
            speak_fn(ans)

        # Rating / RLHF online update
        try:
            rating_raw = input("Rate (1-5, Enter để bỏ qua)> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if rating_raw in {"1", "2", "3", "4", "5"}:
            try:
                note = input("Ghi chú (Enter để bỏ qua)> ").strip()
            except (EOFError, KeyboardInterrupt):
                note = ""

            agent.memory.add_feedback(
                turn_id=turn_id,
                user_query=q,
                assistant_answer=ans,
                reward=int(rating_raw),
                note=note,
            )

            qdap_reward = (int(rating_raw) - 3) / 2.0
            agent.update_qdap_feedback(qdap_reward)
            print(" Phản hồi đã lưu.\n")

if __name__ == "__main__":
    main()
