"""
verify_chunks.py — Check chunk quality after ingest.

Run after ingest.py:
    python verify_chunks.py
"""
import json, sys, os
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

chunks_path = ROOT / "data" / "processed" / "chunks.json"
if not chunks_path.exists():
    print("✗ chunks.json không tồn tại. Chạy ingest.py trước.")
    sys.exit(1)

chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
sizes  = [len(c["text"]) for c in chunks]

print("=" * 60)
print("CHUNK QUALITY REPORT")
print("=" * 60)

#  Basic stats 
print(f"\n Tổng số chunks  : {len(chunks)}")
print(f"   Min size        : {min(sizes)} chars")
print(f"   Avg size        : {sum(sizes)//len(sizes)} chars")
print(f"   Max size        : {max(sizes)} chars")

#  Size distribution 
buckets = [
    ("<100",  lambda s: s < 100),
    ("100–300", lambda s: 100 <= s < 300),
    ("300–500", lambda s: 300 <= s < 500),
    ("500–700", lambda s: 500 <= s < 700),
    ("700–900", lambda s: 700 <= s < 900),
    (">900",  lambda s: s >= 900),
]
print("\n Phân phối kích thước:")
for label, fn in buckets:
    count = sum(1 for s in sizes if fn(s))
    bar   = "█" * (count * 40 // len(sizes)) if len(sizes) else ""
    flag  = " WARNING:" if label in (">900", "<100") and count > 0 else ""
    print(f"   {label:10s}: {count:4d}  {bar}{flag}")

# Per-source 
sources = Counter(c["source"] for c in chunks)
print("\n Chunks per PDF:")
for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"   {cnt:4d}  {src}")

#  Oversized chunks detail 
oversized = [c for c in chunks if len(c["text"]) > 900]
if oversized:
    print(f"\  Oversized chunks (>900 chars): {len(oversized)}")
    for c in oversized[:5]:
        print(f"   {len(c['text'])} chars | {c['source']} | "
              f"Điều {c.get('article_number','?')} | {c['text'][:80]!r}")

#  Key query tests 
TEST_QUERIES = [
    "học phần chung",
    "điều kiện tốt nghiệp",
    "học phí",
    "cảnh báo học tập",
    "tín chỉ tối",          
]

print("\n Keyword coverage test:")
for q in TEST_QUERIES:
    hits = [c for c in chunks if q.lower() in c["text"].lower()]
    flag = "✓" if hits else "✗"
    print(f"   {flag} '{q}': {len(hits)} chunk(s)", end="")
    if hits:
        best = max(hits, key=lambda c: len(c["text"]))
        idx  = best["text"].lower().find(q.lower())
        print(f" | best: {best['text'][idx:idx+120]!r}", end="")
    print()

#  Overlap check 
print("\n Overlap check (chunks should share partial text with neighbours):")
stsv = [c for c in chunks if "STSV" in c["source"]]
overlap_count = 0
for i in range(1, min(len(stsv), 20)):
    a_end = stsv[i-1]["text"][-100:]
    b_start = stsv[i]["text"][:100]
    # Rough overlap: any 20-char substring of a_end in b_start
    words_a = set(stsv[i-1]["text"][-150:].split())
    words_b = set(stsv[i]["text"][:150].split())
    shared  = words_a & words_b
    if len(shared) >= 3:
        overlap_count += 1
print(f"   {overlap_count}/19 consecutive STSV chunk pairs have word overlap (target ≥ 10)")

print("\n" + "=" * 60)
ok = (
    len(oversized) <= 8                                         # allow a few hard sentences
    and sum(1 for s in sizes if s < 100) < len(chunks) * 0.05  # <5% truly tiny
    and sum(1 for q in TEST_QUERIES
            if any(q in c["text"].lower() for c in chunks)) >= 4
)
if ok:
    print(" PASS — chunk quality acceptable for retrieval")
else:
    print(" FAIL — review issues above before using retrieval")
print("=" * 60)
