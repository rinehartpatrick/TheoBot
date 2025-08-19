# TheoBot: A Fine-Tuned LLM for Theological Studies
---
TheoBot is a specialized large language model fine-tuned to provide accurate and contextually aware answers on theology, with a focus on Christianity and Judaism. It is built on the `Qwen/Qwen2.5-7B-Instruct` model and has been aligned with the six core themes of the UK's **GCSE Religious Education** curriculum:

1.  Relationships & Families
2.  Religion & Life
3.  The Existence of God & Revelation
4.  Peace and Conflict
5.  Crime and Punishment
6.  Human Rights & Social Justice

The goal of this project was to create a more reliable and factually grounded assistant for students and enthusiasts in this domain, leveraging the efficiency of LoRA for parameter-efficient fine-tuning.

## Key Features

*   **Specialized Knowledge**: Trained on ~1,700 examples covering the GCSE RE curriculum.
*   **Efficient Training**: Uses **Low-Rank Adaptation (LoRA)**, resulting in a small, portable adapter (~100-300MB) that can be applied to the base model.
*   **High Performance**: Built on the powerful `Qwen/Qwen2.5-7B-Instruct` base model.
*   **ROCm Compatible**: The entire training and inference pipeline is tested and validated on the AMD ROCm stack.

---

## Evaluation & Results

The model's performance was rigorously benchmarked against the original base model across several domains. The results demonstrate a clear and positive impact from the fine-tuning process.

### 1. Overall Performance Comparison

This chart provides a high-level view of model performance across different evaluation sets. It clearly shows the trade-offs and successes of the fine-tuning process.

<br>
<img src="https://github.com/rinehartpatrick/TheoBot/blob/main/Simplified%20Bot%20Comparison%20by%20Domain%20Dataset%20(Accuracy%20%25).png?raw=true" width="600" height="400">
<br>

**Analysis**:
*   **Success in Generalization**: The model achieves an exceptional **99.2%** on "Out of domain Base" questions, confirming that the fine-tuning did not harm the model's excellent general-purpose reasoning and instruction-following abilities.
*   **Targeted Improvement**: While performance drops in the more specialized theological domains, the fine-tuned model consistently outperforms the base model in these areas (as shown in the next section's graphs).
*   **Areas for Improvement**: The lowest score is in the "Nwar domain Base" category, suggesting that the model struggles most with nuanced questions that are adjacent to but not directly covered by its training data. This highlights a potential area for future data curation.

### 2. Performance Heatmap by Rubric

This heatmap visualizes the model's strengths and weaknesses across our evaluation rubric for each question category.

<br>
<img src="https://github.com/rinehartpatrick/TheoBot/blob/main/Heatmap%20Performance%20by%20Question%20Category%20and%20Rubric.png?raw=true" width="600" height="400">
<br>

**Analysis**:
*   **Core Competencies**: The model consistently scores high in **Task Understanding (TU)** and **Clarity (CL)** across all domains. It understands what is being asked and provides clear, well-structured answers.
*   **Pinpointed Weaknesses**: The primary areas of weakness, especially in the theological domains, are **Scriptural Fidelity (SF)**, **Thematic Reasoning (TR)**, and **Specificity (SS)**. This indicates that while the model can formulate a good answer, its ability to recall specific scriptural details and provide deep, well-supported reasoning is the most challenging task.
*   **Perspective Fit (PF)**: The model shows moderate success in adopting a specific theological perspective when asked, but this remains a difficult skill that requires precise terminology and understanding.

### 3. Distribution of Errors (In-Domain Scripture Tasks)

This chart is crucial as it breaks down the *types* of errors made on the most critical in-domain tasks, revealing the direct impact of fine-tuning.

<br>
<img src="" width="600" height="400">
<br>

**Analysis**:
The most significant finding from our evaluation is related to **Scriptural Fidelity (SF)**. A staggering **76%** of the fine-tuned model's errors on in-domain tasks were related to either citing a completely wrong verse (40%) or misinterpreting the correct one (36%). This was the single biggest failure mode of the base model. The fine-tuning process was specifically designed to address this, and while it remains a challenge, the significant reduction in these types of errors (as shown in other charts) is a major success of this project.

---

## Technical Deep Dive

### Methodology and Technical Stack

*   **Technique**: Supervised Fine-Tuning (SFT) with **Low-Rank Adaptation (LoRA)**. We chose LoRA for its efficiency, as it trains only small low-rank matrices on top of a frozen base model. This allows for fast training and produces a small, portable adapter (~100–300MB) while preserving the base model’s general abilities.
*   **Precision**: We used **`bf16`** weights for the base model. An initial attempt at QLoRA on ROCm proved unstable due to `bitsandbytes` library issues, so we reverted to the more stable `bf16` implementation.
*   **Hardware Stack**: AMD ROCm (tested on Radeon RX 7900 XTX) on Ubuntu.
*   **Core Libraries**: `transformers`, `peft`, `datasets`, and `accelerate`.

### Key Training Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Base model** | `Qwen/Qwen2.5-7B-Instruct` | |
| **LoRA config** | `r=16`, `lora_alpha=32`, `dropout=0.05` | Applied to `q_proj`, `k_proj`, `v_proj`, `o_proj` attention layers. |
| **Batching** | `per_device_train_batch_size=1`, `gradient_accumulation_steps=16` | Effective batch size of 16. |
| **Precision**| `bf16=True`, `attn_implementation="sdpa"` | |
| **Scheduler**| `cosine`, `warmup_ratio=0.03` | |
| **LR / Epochs**| `learning_rate=2e-4`, `num_train_epochs=3` | |
| **Output** | Adapter saved to `~/Theobot/adapter/` | |

### Data Format

The training data is a `.jsonl` file where each line is a JSON object.

```json
{"prompt": "User: <question>\nAssistant:", "completion": "<answer>"}
```

---

## Setup and Usage

### Technical Requirements

*   **GPU**: AMD RDNA3 (tested on Radeon RX 7900 XTX, 24GB VRAM).
*   **RAM**: ≥ 16GB recommended.
*   **Disk**: ~25–30GB free (for model, cache, and outputs).
*   **OS / Drivers**: Ubuntu 24.04 LTS with a working ROCm 6.4 installation.

### Environment Setup

1.  **Create and activate a Python virtual environment:**
    ```bash
    sudo apt install -y python3-venv
    python3 -m venv ~/theobot-venv
    source ~/theobot-venv/bin/activate
    ```

2.  **Install ROCm PyTorch and required libraries:**
    ```bash
    pip install --upgrade pip wheel setuptools
    pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch torchvision torchaudio
    pip install "transformers>=4.43" "accelerate>=0.31" peft datasets sentencepiece huggingface_hub tqdm
    ```

3.  **Set recommended environment variables:**
    ```bash
    # Use a clean cache location to avoid permission issues
    export HF_HOME=~/hf_cache

    # HIP allocator tweak to reduce memory fragmentation on ROCm
    export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
    ```

### Training

The training script skeleton is provided below. Save your full script as `~/train_qlora.py` and run it.

```python
# ~/train_qlora.py (Skeleton)
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                          Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model
import torch, os, pathlib

# --- Constants ---
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_FILE  = str(pathlib.Path.home() / "Desktop" / "train_fixed.jsonl")
OUTPUT_DIR = str(pathlib.Path.home() / "Theobot")

# --- Load and Prepare Dataset ---
# (Your dataset loading and tokenization logic here)

# --- Load Model and LoRA Config ---
model = AutoModelForCausalLM.from_pretrained(...)
lora_config = LoraConfig(r=16, lora_alpha=32, ...)
model = get_peft_model(model, lora_config)

# --- Training Arguments and Trainer ---
training_args = TrainingArguments(output_dir=OUTPUT_DIR, per_device_train_batch_size=1, ...)
trainer = Trainer(model=model, args=training_args, train_dataset=ds, ...)

# --- Train and Save ---
trainer.train()
save_path = os.path.join(OUTPUT_DIR, "adapter")
model.save_pretrained(save_path)
print("Saved adapter to:", save_path)
```

**To run the training:**

```bash
source ~/theobot-venv/bin/activate
python ~/train_qlora.py
```

### Inference (Single Prompt Smoke Test)

Use the following Python script to test your trained adapter.

```python
# inference.py
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = os.path.expanduser("~/Theobot/adapter")

# Load base model and tokenizer
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True); tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")

# Merge LoRA adapter
m = PeftModel.from_pretrained(m, ADAPTER); m.eval()

# Run inference
p = "User: What is the main message of Deuteronomy 24:17?\nAssistant:"
x = tok(p, return_tensors="pt").to(m.device)
with torch.inference_mode():
    y = m.generate(**x, max_new_tokens=150, temperature=0.0, do_sample=False, pad_token_id=tok.eos_token_id)
print(tok.decode(y[0], skip_special_tokens=True))
```

---

## Troubleshooting

*   **HIP “Module not initialized”**: Ensure your user is in the `video` and `render` groups; `rocminfo` must show your GPU. A reboot is often required after group changes.
*   **Permission errors in cache**: Use a fresh cache (`export HF_HOME=~/hf_cache`) or run `chown -R "$USER:$USER" ~/.cache/huggingface`.
*   **OOM (Out of Memory)**: Lower `max_new_tokens` during inference, or increase `gradient_accumulation_steps` during training.

---

## Appendix: Evaluation Rubric

Each answer was scored from 0–2 on the following six criteria.

| Criterion | 2 (Excellent) | 1 (Partial) | 0 (Failure) |
| :--- | :--- | :--- | :--- |
| **Task Understanding (TU)** | Directly answers the question asked. | Partially answers; some drift or missed part. | Off-topic or dodges the question. |
| **Factual/Scriptural Fidelity (SF)** | Factually correct; reflects the correct scripture. | Mostly right but with a minor error or conflation. | Incorrect core claim or wrong verse. |
| **Thematic Reasoning (TR)** | Explains the "why" clearly with sound reasoning. | Mentions theme but reasoning is weak or shallow. | No real reasoning; just a statement. |
| **Perspective Fit (PF)\*** | Correct tradition framing and terminology. | Mostly right but imprecise terminology. | Uses wrong tradition or misrepresents it. |
| **Specificity & Support (SS)** | Uses concrete details and relevant examples. | Some specifics, but mostly generic. | Vague/generalities only. |
| **Clarity & Structure (CL)** | Clear, concise, and well-organized. | Understandable but wordy or disorganized. | Hard to follow. |

*\*PF is only scored if the question specifies a tradition (e.g., "from a Judaism perspective").*
