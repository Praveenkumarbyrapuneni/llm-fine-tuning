import json
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("data/training-ready.jsonl")
VALID_LABELS = {"positive", "negative", "neutral"}
SAMPLE_COUNT = 3


def extract_label(text: str) -> str | None:
    # Qwen3 wraps assistant reply in <think>...</think> before the actual label
    # Pattern: <|im_start|>assistant\n<think>\n\n</think>\n\nneutral<|im_end|>
    marker = "<|im_start|>assistant\n"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    after = text[idx + len(marker):]

    # skip past </think> block if present
    think_end = after.find("</think>")
    if think_end != -1:
        after = after[think_end + len("</think>"):]

    label = after.split("<")[0].strip().lower()
    return label if label in VALID_LABELS else None


def validate() -> None:
    if not INPUT_PATH.exists():
        print(f"File not found: {INPUT_PATH}")
        print("Run dwn-train-ready.py first.")
        return

    rows = INPUT_PATH.read_text(encoding="utf-8").splitlines()
    total = len(rows)
    print(f"Total rows in file: {total:,}\n")

    label_counts: Counter = Counter()
    broken = []

    print(f"--- {SAMPLE_COUNT} Sample Rows ---\n")

    for i, line in enumerate(rows):
        try:
            text = json.loads(line)["text"]
        except (json.JSONDecodeError, KeyError):
            broken.append(i)
            continue

        label = extract_label(text)
        if label is None:
            broken.append(i)
            continue

        label_counts[label] += 1

        if i < SAMPLE_COUNT:
            print(f"Row {i + 1}:")
            print(text)
            print("-" * 60)

    valid = total - len(broken)

    print("\n--- Label Distribution ---\n")
    for label in VALID_LABELS:
        count = label_counts[label]
        pct = (count / valid * 100) if valid else 0
        flag = "  ⚠ LOW" if pct < 20 else ""
        print(f"  {label:<10}: {count:>6,}  ({pct:.1f}%){flag}")

    print(f"\n--- Summary ---\n")
    print(f"  Valid rows   : {valid:,}")
    print(f"  Broken rows  : {len(broken):,}")

    if broken:
        print(f"  Broken at lines: {broken[:10]} {'...' if len(broken) > 10 else ''}")
        print("\n  Fix dwn-train-ready.py and rerun.")
    else:
        print("\n  All rows valid. Ready for training.")


if __name__ == "__main__":
    validate()
