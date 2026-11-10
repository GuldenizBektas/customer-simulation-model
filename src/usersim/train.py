"""QLoRA SFT of the customer simulator with Unsloth + TRL.

Runs on a GPU (Kaggle T4). All heavy imports live inside functions so this
module can be imported (and ``--help`` works) on a laptop without torch.

Loss is computed ONLY on assistant (= customer) tokens via
``train_on_responses_only``: the model learns to talk like the customer,
not to imitate the agent or memorise the scenario prompt.

Usage (on Kaggle):
    python -m usersim.train --config configs/train.yaml
    python -m usersim.train --config configs/train.yaml --max-examples 500 --epochs 1  # smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_model(cfg: dict):
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_len"],
        load_in_4bit=cfg.get("load_in_4bit", True),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template=cfg["chat_template"])
    return model, tokenizer


def load_split(path: str, tokenizer, max_examples: int | None = None):
    """JSONL with a "messages" column -> dataset with a rendered "text" column."""
    from datasets import load_dataset

    ds = load_dataset("json", data_files=path, split="train")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    render = lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
    return ds.map(render, remove_columns=ds.column_names)


def train(cfg: dict) -> str:
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    model, tokenizer = load_model(cfg)
    train_ds = load_split(cfg["train_file"], tokenizer, cfg.get("max_examples"))
    eval_ds = load_split(cfg["dev_file"], tokenizer, cfg.get("max_eval_examples"))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=cfg["max_seq_len"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["lr"],
            warmup_ratio=cfg["warmup_ratio"],
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            optim="adamw_8bit",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=cfg["eval_steps"],
            save_steps=cfg["save_steps"],
            output_dir=cfg["output_dir"],
            seed=cfg["seed"],
            report_to="none",
        ),
    )
    # Mask system + agent tokens: only customer turns contribute to the loss.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.train()

    adapter_dir = str(Path(cfg["output_dir"]) / "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    if cfg.get("export_gguf"):
        model.save_pretrained_gguf(
            str(Path(cfg["output_dir"]) / "gguf"), tokenizer, quantization_method=cfg["gguf_quant"]
        )
    if cfg.get("hub_repo"):
        model.push_to_hub(cfg["hub_repo"], private=True)  # uses HF_TOKEN from the environment
        tokenizer.push_to_hub(cfg["hub_repo"], private=True)
    return adapter_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="QLoRA SFT of the customer simulator (GPU only).")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--max-examples", type=int, help="subsample train set (smoke test)")
    parser.add_argument("--epochs", type=float, help="override epochs")
    parser.add_argument("--export-gguf", action="store_true", help="also write a GGUF for Ollama")
    args = parser.parse_args(argv)
    import yaml

    cfg = yaml.safe_load(open(args.config))
    if args.max_examples:
        cfg["max_examples"] = args.max_examples
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.export_gguf:
        cfg["export_gguf"] = True
    print("adapter saved to", train(cfg))


if __name__ == "__main__":
    main()
