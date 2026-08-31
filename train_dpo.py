#!/usr/bin/env python3
"""Stage 2: DPO on the SFT model, using (chosen, rejected) move pairs.

Runs on Apple Silicon via PyTorch MPS with bf16 LoRA (bitsandbytes 4-bit is
CUDA-only; a 7B bf16 base + LoRA fits comfortably in 64GB unified memory).
Prereq: pip install torch transformers trl peft datasets

NOTE: this trains from the HF base + SFT LoRA re-applied, i.e. point
--base at the original HF repo and --sft-adapter at an SFT adapter trained
with PEFT -- OR simply point --base at a HF-format export of your fused SFT
model. Untested script: expect to adjust versions/arg names to your installed
trl release.

After training, convert to MLX for the proposer:
  mlx_lm.convert --hf-path model_dpo --mlx-path model_dpo_mlx
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="HF model (ideally your SFT-fused model in HF format)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="model_dpo")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16).to(device)

    ds = load_dataset("json", data_files={
        "train": f"{args.data}/dpo_train.jsonl",
        "eval": f"{args.data}/dpo_valid.jsonl"})

    peft_cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj",
                                          "o_proj", "gate_proj",
                                          "up_proj", "down_proj"])
    cfg = DPOConfig(output_dir=args.out, per_device_train_batch_size=args.batch,
                    gradient_accumulation_steps=8, num_train_epochs=args.epochs,
                    learning_rate=args.lr, beta=args.beta, bf16=False,
                    logging_steps=20, eval_strategy="steps", eval_steps=200,
                    save_strategy="epoch", report_to=[])
    trainer = DPOTrainer(model=model, args=cfg, processing_class=tok,
                         train_dataset=ds["train"], eval_dataset=ds["eval"],
                         peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved {args.out}; convert with mlx_lm.convert for the proposer")


if __name__ == "__main__":
    main()
