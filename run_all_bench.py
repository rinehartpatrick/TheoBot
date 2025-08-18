# run_all_bench.py
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ===== CONFIG =====
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"            # the base model you trained on
ADAPTER_PATH = str(Path.home() / "Theobot" / "adapter")  # your LoRA adapter
MAX_NEW_TOKENS = 180
TEMPERATURE = 0.0   # deterministic, fair comparison
DO_SAMPLE = False

DESKTOP = Path.home() / "Desktop"
INPUTS = {
    "in_domain": DESKTOP / "in_domain_questions.txt",
    "near_domain": DESKTOP / "near_domain_questions.txt",
    "out_of_domain": DESKTOP / "out_of_domain_questions.txt",
}
OUTDIR = DESKTOP / "bench_results"
OUTDIR.mkdir(parents=True, exist_ok=True)
# ===================

def load_base():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tok, model

def load_theobot():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return tok, model

def answer_file(tok, model, infile: Path, outfile: Path):
    assert infile.exists(), f"Missing input file: {infile}"
    qs = [q.strip() for q in infile.read_text(encoding="utf-8").splitlines() if q.strip()]
    with outfile.open("w", encoding="utf-8") as out:
        for i, q in enumerate(qs, 1):
            prompt = f"User: {q}\nAssistant:"
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=DO_SAMPLE,
                    top_p=1.0,
                    pad_token_id=tok.eos_token_id,
                )
            ans = tok.decode(outputs[0], skip_special_tokens=True)
            out.write(f"Q{i}: {q}\nA{i}: {ans}\n\n")
            print(f"[{outfile.name}] {i}/{len(qs)}")

def main():
    # 1) Run BASE model on all three sets (keeps VRAM use reasonable)
    print("Loading BASE model…")
    tok, model = load_base()
    for name, infile in INPUTS.items():
        outfile = OUTDIR / f"{name}_base.txt"
        print(f"Running BASE on {name} → {outfile}")
        answer_file(tok, model, infile, outfile)
    del tok, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2) Run THEOBOT (base + adapter) on all three sets
    print("Loading THEOBOT (LoRA adapter)…")
    tok, model = load_theobot()
    for name, infile in INPUTS.items():
        outfile = OUTDIR / f"{name}_theobot.txt"
        print(f"Running THEOBOT on {name} → {outfile}")
        answer_file(tok, model, infile, outfile)

    print(f"\nDone. Results are in: {OUTDIR}")

if __name__ == "__main__":
    # Optional: if you have multiple GPUs, choose one explicitly:
    # os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    main()

