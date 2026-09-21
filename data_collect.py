# Script to extract CUAD contract data + clause annotations (for agentic RAG)
#
# Downloads:
#   1. Raw contract text from dvgodoy/CUAD_v1_Contract_Understanding_PDF (if not cached)
#   2. CUAD-QA annotations from The Atticus Project official repository (41 clause categories)
#
# Outputs:
#   - data/contracts/*.txt            — raw contract text files
#   - data/contracts_annotations.json — {filename: [{category, start_char, end_char, text}]}

import io
import json
import re
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

CONTRACTS_DIR = Path("data/contracts")
ANNOTATIONS_FILE = Path("data/contracts_annotations.json")
CUAD_ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"

# ---------------------------------------------------------------------------
# 1. Download raw contract text files (if not already downloaded)
# ---------------------------------------------------------------------------

CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
existing_contracts = list(CONTRACTS_DIR.glob("*.txt"))

if len(existing_contracts) >= 500:
    print(
        f"Found {len(existing_contracts)} existing contract text files in {CONTRACTS_DIR}, skipping download."
    )
else:
    print("Downloading contracts from dvgodoy/CUAD_v1_Contract_Understanding_PDF...")
    from datasets import load_dataset

    contracts = load_dataset("dvgodoy/CUAD_v1_Contract_Understanding_PDF")
    print(f"Loaded {len(contracts['train'])} contracts.")
    for row in contracts["train"]:
        fname = row["file_name"].replace(".pdf", ".txt")
        with open(CONTRACTS_DIR / fname, "w", encoding="utf-8") as f:
            f.write(row["text"])
    print(f"Wrote {len(contracts['train'])} contract text files to {CONTRACTS_DIR}")
    existing_contracts = list(CONTRACTS_DIR.glob("*.txt"))

# Build index of existing filenames on disk
existing_filenames = {f.name for f in existing_contracts}
# Build normalized map: normalized_name -> actual_filename
normalized_map = {}
for fname in existing_filenames:
    norm = unicodedata.normalize("NFKD", fname).strip()
    norm = re.sub(r"\s+", " ", norm)
    normalized_map[norm] = fname

# ---------------------------------------------------------------------------
# 2. Download CUAD-QA master annotations (41 categories)
# ---------------------------------------------------------------------------

print(f"\nDownloading CUAD master annotations from {CUAD_ZIP_URL}...")
req = urllib.request.Request(CUAD_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=60) as resp:
    zip_bytes = resp.read()

print(f"Downloaded {len(zip_bytes):,} bytes. Extracting CUADv1.json...")
with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z, z.open("CUADv1.json") as f:
    cuad = json.load(f)

print(f"Loaded CUADv1 dataset with {len(cuad['data'])} contract documents.")

# Build annotations mapping: {filename.txt: [{category, start_char, end_char, text}]}
annotations = defaultdict(list)
total_spans = 0
matched_docs = 0
unmatched_docs = 0

for doc in cuad["data"]:
    title = doc.get("title", "")
    target_filename = f"{title}.txt"

    # Match to actual file on disk
    matched_filename = None
    if target_filename in existing_filenames:
        matched_filename = target_filename
    else:
        norm_title = unicodedata.normalize("NFKD", f"{title}.txt").strip()
        norm_title = re.sub(r"\s+", " ", norm_title)
        if norm_title in normalized_map:
            matched_filename = normalized_map[norm_title]
        else:
            # Fallback: substring matching
            for norm_key, real_name in normalized_map.items():
                if norm_key[:40] == norm_title[:40]:
                    matched_filename = real_name
                    break

    if not matched_filename:
        unmatched_docs += 1
        continue

    matched_docs += 1

    for paragraph in doc.get("paragraphs", []):
        for qa in paragraph.get("qas", []):
            qa_id = qa.get("id", "")
            # Extract category: "{title}__{Category}"
            parts = qa_id.split("__")
            if len(parts) >= 2:
                category = parts[1]
            else:
                # Extract from question: '...related to "{Category}"'
                q_text = qa.get("question", "")
                m = re.search(r'"([^"]+)"', q_text)
                category = m.group(1) if m else "Unknown"

            for answer in qa.get("answers", []):
                ans_text = answer.get("text", "").strip()
                ans_start = answer.get("answer_start")
                if ans_text and ans_start is not None:
                    annotations[matched_filename].append(
                        {
                            "category": category,
                            "start_char": ans_start,
                            "end_char": ans_start + len(ans_text),
                            "text": ans_text[
                                :200
                            ],  # truncate snippet for storage efficiency
                        }
                    )
                    total_spans += 1

# Write annotations to JSON
with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
    json.dump(dict(annotations), f, indent=2, ensure_ascii=False)

print("\nAnnotation extraction complete:")
print(f"  Matched contracts:   {matched_docs}/{len(cuad['data'])}")
print(f"  Unmatched:           {unmatched_docs}")
print(f"  Total clause spans:  {total_spans:,}")
print(f"  Contracts annotated: {len(annotations)}")
print(f"  Output saved to:     {ANNOTATIONS_FILE}")

# Print category distribution
cat_counts = defaultdict(int)
for file_anns in annotations.values():
    for ann in file_anns:
        cat_counts[ann["category"]] += 1

print("\nTop 20 categories by clause count:")
for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  {cat:40s} {count:5d}")
