import os
os.environ["USE_TF"] = "0"

import json
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME   = "Qwen/Qwen3-1.7B"
ADAPTER_PATH = "adapters/sentiment"
TEST_PATH    = Path("data/test-ready.jsonl")

# 500 rows on laptop takes ~20 minutes. Set to None for all 15k (hours).
MAX_EVAL_ROWS = 500

VALID_LABELS = {"positive", "negative", "neutral"}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_true_label(text: str) -> str | None:
    marker = "<|im_start|>assistant\n"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    after = text[idx + len(marker):]
    think_end = after.find("</think>")
    if think_end != -1:
        after = after[think_end + len("</think>"):]
    label = after.split("<")[0].strip().lower()
    return label if label in VALID_LABELS else None


def extract_prompt(text: str) -> str | None:
    # Everything up to and including the assistant marker — model completes from here
    marker = "<|im_start|>assistant\n"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    return text[:idx + len(marker)]


def extract_predicted_label(generated: str) -> str | None:
    # First try: after </think> block
    think_end = generated.find("</think>")
    if think_end != -1:
        after = generated[think_end + len("</think>"):]
        label = after.split("<")[0].strip().lower()
        if label in VALID_LABELS:
            return label
    # Fallback: find any valid label anywhere in the output
    match = re.search(r"\b(positive|negative|neutral)\b", generated, re.IGNORECASE)
    return match.group(1).lower() if match else None


def evaluate() -> None:
    device = get_device()
    print(f"Device      : {device}")
    print(f"Adapter     : {ADAPTER_PATH}")
    print(f"Test file   : {TEST_PATH}")

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.to(device)
    model.eval()
    print("Model loaded.\n")

    rows = TEST_PATH.read_text(encoding="utf-8").splitlines()
    if MAX_EVAL_ROWS:
        rows = rows[:MAX_EVAL_ROWS]
    print(f"Evaluating on {len(rows):,} rows...\n")

    correct = 0
    total   = 0
    skipped = 0
    per_label: dict = {
        "positive": {"correct": 0, "total": 0},
        "negative": {"correct": 0, "total": 0},
        "neutral":  {"correct": 0, "total": 0},
    }

    for i, line in enumerate(rows):
        text = json.loads(line)["text"]
        true_label = extract_true_label(text)
        prompt     = extract_prompt(text)

        if not true_label or not prompt:
            skipped += 1
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )
        pred_label = extract_predicted_label(generated)

        if pred_label:
            total += 1
            match = pred_label == true_label
            if match:
                correct += 1
            per_label[true_label]["total"]   += 1
            per_label[true_label]["correct"] += (1 if match else 0)
        else:
            skipped += 1

        if (i + 1) % 50 == 0:
            running_acc = correct / total * 100 if total else 0
            print(f"  [{i + 1:>4}/{len(rows)}]  running accuracy: {running_acc:.1f}%")

    print(f"\n{'='*40}")
    print(f"RESULTS — {total} rows evaluated, {skipped} skipped")
    print(f"{'='*40}")
    overall = correct / total * 100 if total else 0
    print(f"\nOverall accuracy : {correct}/{total} = {overall:.1f}%\n")
    print("Per-label breakdown:")
    for label, counts in per_label.items():
        if counts["total"] > 0:
            pct = counts["correct"] / counts["total"] * 100
            bar = "✅" if pct >= 80 else "⚠️ " if pct >= 65 else "❌"
            print(f"  {bar}  {label:<10}: {counts['correct']:>3}/{counts['total']:>3} = {pct:.1f}%")


if __name__ == "__main__":
    evaluate()
