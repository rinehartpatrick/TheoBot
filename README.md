# TheoBot
Excellent, this document provides a comprehensive technical overview of the fine-tuning process. Based on this information, here is a structured breakdown of the methodology, implementation, and usage instructions.

### Executive Summary

This project involved the **Supervised Fine-Tuning (SFT)** of the `Qwen/Qwen2.5-7B-Instruct` large language model to create a specialized version, "Theobot." The goal was to align the model with the six themes of the GCSE Religious Education curriculum. This was achieved using **Low-Rank Adaptation (LoRA)**, a parameter-efficient fine-tuning (PEFT) method, which allows for rapid training and produces a small, portable adapter file (~100-300MB) while preserving the base model's extensive general knowledge. The entire process was conducted on an AMD ROCm hardware stack.

---

### 1. Methodology and Technical Stack

#### Fine-Tuning Strategy

*   **Technique**: Supervised Fine-Tuning (SFT) was used to train the model on a dataset of question-answer pairs.
*   **Parameter-Efficient Fine-Tuning (PEFT)**: The project employed **LoRA (Low-Rank Adaptation)**. This was a strategic choice because:
    *   **Efficiency**: It only trains a small number of new weights (low-rank matrices) instead of the entire model.
    *   **Speed**: Training is significantly faster than a full fine-tune.
    *   **Portability**: The output is a small adapter file, not a full model fork.
    *   **Preservation**: It maintains the strong general capabilities of the original `Qwen2` model.
*   **Precision**: The training was performed using **`bf16`** (BFloat16) precision. An initial attempt using QLoRA (a quantized version of LoRA) was abandoned due to instability with the `bitsandbytes` library on the ROCm platform. The `bf16` approach proved more stable and reliable.

#### Technical Stack

| Component | Specification |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-7B-Instruct` from Hugging Face |
| **Hardware** | AMD Radeon RX 7900 XTX GPU (24GB VRAM) |
| **OS / Driver** | Ubuntu 24.04 LTS with AMD ROCm 6.4 |
| **Core Libraries** | `transformers`, `peft`, `datasets`, `accelerate`, PyTorch for ROCm |

---

### 2. Training Implementation

#### Dataset and Format

*   **File**: The training data was consolidated into a single `train_fixed.jsonl` file.
*   **Structure**: Each line in the file is a JSON object containing a prompt and a corresponding completion, designed for instruction-following.
    ```json
    {"prompt": "User: <question>\nAssistant:", "completion": "<answer>"}
    ```

#### Key Training Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **LoRA `r` (Rank)** | `16` | The rank of the low-rank matrices. |
| **LoRA `lora_alpha`** | `32` | A scaling factor for the LoRA weights. |
| **LoRA `target_modules`** | `["q_proj","k_proj","v_proj","o_proj"]` | The specific attention layers to which LoRA was applied. |
| **Batch Size** | `1` (per device) | The number of samples processed per device at once. |
| **Grad. Accumulation** | `16` | Steps to accumulate gradients before an optimizer step (Effective Batch Size = 16). |
| **Learning Rate** | `2e-4` | The initial learning rate for the optimizer. |
| **Epochs** | `3` | The total number of times the training dataset was passed through the model. |
| **LR Scheduler** | `cosine` | The learning rate schedule, with a warmup ratio of 3%. |
| **Precision** | `bf16=True` | Enabled 16-bit brain floating-point precision for training. |

#### Training Script Skeleton

The training process was managed by a Python script utilizing the Hugging Face `Trainer` API. The key steps are:

1.  **Load Data**: The `train_fixed.jsonl` file is loaded using the `datasets` library.
2.  **Format and Tokenize**: The prompt and completion are merged into a single text field, which is then tokenized for the model.
3.  **Load Model & Tokenizer**: The base `Qwen/Qwen2.5-7B-Instruct` model is loaded in `bfloat16` precision.
4.  **Apply LoRA Config**: The LoRA configuration is defined and applied to the base model using `get_peft_model`.
5.  **Set Training Arguments**: All hyperparameters (learning rate, batch size, epochs, etc.) are defined in a `TrainingArguments` object.
6.  **Instantiate and Run Trainer**: A `Trainer` instance is created with the model, arguments, and dataset, and `trainer.train()` is called to start the fine-tuning process.
7.  **Save Adapter**: The final trained LoRA adapter is saved to the specified output directory.

```python
# --- Key Sections of the Training Script ---

# 1. Load and prepare the dataset
raw = load_dataset("json", data_files=DATA_FILE, split="train")
ds = raw.map(lambda e: {"text": f"{e['prompt']}{e['completion']}"}, ...)
ds = ds.map(lambda b: tok(b["text"], ...), batched=True)

# 2. Load the base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", ...
)

# 3. Define and apply the LoRA adapter
lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", ...], ...)
model = get_peft_model(model, lora)

# 4. Define training arguments and run the trainer
args = TrainingArguments(output_dir=OUTPUT_DIR, learning_rate=2e-4, ...)
trainer = Trainer(model=model, args=args, train_dataset=ds, ...)
trainer.train()

# 5. Save the final adapter
model.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
```

---

### 3. Usage and Evaluation

#### Running Inference (Single Prompt Test)

To test the fine-tuned model, the base model is first loaded, and then the trained LoRA adapter is merged on top.

**Workflow:**
1.  Load the tokenizer and the base `Qwen2` model in `bfloat16`.
2.  Load the LoRA adapter from the output directory (`~/Theobot/adapter/`).
3.  Combine the adapter and base model using `PeftModel.from_pretrained()`.
4.  Tokenize a prompt and pass it to `model.generate()` to get a response.

```python
# --- Inference Script Example ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Define paths
BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = os.path.expanduser("~/Theobot/adapter")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")

# Apply the LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# Prepare and run a prompt
prompt = "User: What is the main message of Deuteronomy 24:17?\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### Benchmarking Process

A systematic evaluation was performed by generating responses from both the base model and the fine-tuned "Theobot" across three distinct question sets:
*   `in_domain_questions.txt`
*   `near_domain_questions.txt`
*   `out_of_domain_questions.txt`

The provided `run_all_bench.py` script automates this process, producing six output files that allow for a direct comparison of the models' performance in each domain.

---

### 4. Troubleshooting Quick Guide

*   **HIP "Module not initialized" Error**: This is a ROCm driver issue. Ensure your user is in the `video` and `render` groups and that `rocminfo` correctly identifies your GPU. A reboot is often required after making changes.
*   **Cache Permission Errors**: To avoid issues with shared Hugging Face cache directories, use a local cache by setting the environment variable: `export HF_HOME=~/hf_cache`.
*   **Out-of-Memory (OOM) Errors**: During training, increase `gradient_accumulation_steps`. During inference, reduce `max_new_tokens` in the generation call.
*   **Generation Warnings**: If you see harmless warnings about `temperature` or `top_k` when `do_sample=False`, you can safely ignore them or suppress them by setting the environment variable: `export TRANSFORMERS_VERBOSITY=error`.
