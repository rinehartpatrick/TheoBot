from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                          Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model
import torch, os, pathlib

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_FILE  = str(pathlib.Path.home() / "Desktop" / "train_fixed.jsonl")
OUTPUT_DIR = str(pathlib.Path.home() / "Theobot")

raw = load_dataset("json", data_files=DATA_FILE, split="train")
ds = raw.map(lambda e: {"text": f"{e['prompt']}{e['completion']}"}, remove_columns=raw.column_names)

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tok.pad_token = tok.eos_token

ds = ds.map(lambda b: tok(b["text"], truncation=True, padding="max_length", max_length=1024),
            batched=True, remove_columns=["text"])

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True, attn_implementation="sdpa"
)
model.config.use_cache = False
if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["q_proj","k_proj","v_proj","o_proj"],
                  bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=20,
    save_strategy="epoch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="none",
)
trainer = Trainer(model=model, args=args, train_dataset=ds,
                  data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
trainer.train()

save_path = os.path.join(OUTPUT_DIR, "adapter")
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
print("Saved adapter to:", save_path)
