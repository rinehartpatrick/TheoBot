# TheoBot
A fine tuned model of Qwen/Qwen2.5-7B-Instruct using LoRA adapters, on GSCE Theology course, focusing on Christanity and Judiasm.
Technique

Supervised Fine-Tuning (SFT) with LoRA.
We fine-tuned Qwen/Qwen2.5-7B-Instruct using LoRA adapters (PEFT) on ~1,700 Q→A examples aligned to the 6 GCSE RE themes (in-domain verse questions + near-domain thematic questions).

Why LoRA? It trains only small low-rank matrices on top of the frozen base model, giving fast training and small output artifacts (adapter ~100–300MB) while preserving the base model’s general ability.

Precision / quantization: Final setup used bf16 weights for the base model (no bitsandbytes), which was the most stable on ROCm. (We initially attempted QLoRA; on ROCm that added brittleness with bitsandbytes, so we switched to plain LoRA bf16.)

Tools

Libraries:

transformers (Trainer API)

peft (LoRA adapters)

datasets (JSONL loading)

accelerate (under the hood via Trainer)

Model Hub: Hugging Face (public base model: Qwen/Qwen2.5-7B-Instruct)

Hardware stack: AMD ROCm (RX 7900 XTX) on Ubuntu

Note: We used the standard transformers.Trainer rather than SFTTrainer. Either works; the Trainer was sufficient for this dataset and objective.

Key Training Parameters

Base model: Qwen/Qwen2.5-7B-Instruct

LoRA config: r=16, lora_alpha=32, lora_dropout=0.05, targets: ["q_proj","k_proj","v_proj","o_proj"]

Batching: per_device_train_batch_size=1, gradient_accumulation_steps=16 (effective batch 16)

Precision: bf16=True, attn_implementation="sdpa"

Scheduler: cosine, warmup_ratio=0.03

LR / Epochs: learning_rate=2e-4, num_train_epochs=3

Stability flags: gradient_checkpointing=True, model.config.use_cache=False, and model.enable_input_require_grads()

Output: adapter saved to ~/Theobot/adapter/

Data & Format

File: train_fixed.jsonl (one JSON object per line)

Fields:
