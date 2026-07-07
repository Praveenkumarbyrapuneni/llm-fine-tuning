import os
os.environ["USE_TF"] = "0"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_NAME  = "Qwen/Qwen3-1.7B"
DATA_PATH   = "data/training-ready.jsonl"
OUTPUT_DIR  = "adapters/sentiment"

# Set to None for full training — set to 200 for laptop smoke test
MAX_ROWS = None

LORA = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str) -> AutoModelForCausalLM:
    if device == "cuda":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb,
            device_map="auto",
        )
        return prepare_model_for_kbit_training(model)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )
    return model.to(device)


def train() -> None:
    device = get_device()
    print(f"Device     : {device}")
    print(f"Max rows   : {MAX_ROWS or 'all'}")

    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(device)
    model = get_peft_model(model, LORA)
    model.print_trainable_parameters()

    print(f"\nLoading dataset: {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    if MAX_ROWS:
        dataset = dataset.select(range(MAX_ROWS))
    print(f"Training on {len(dataset):,} rows")

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        dataset_text_field="text",
        max_length=512,
        bf16=(device == "cuda"),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training...\n")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nAdapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
