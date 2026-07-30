from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from ttlora import (
    TTLoRAConfig,
    TTLoRATarget,
    get_ttlora_model,
    print_trainable_parameters,
    save_ttlora_adapters,
)


def default_gpt2_c_attn_config() -> TTLoRAConfig:
    return TTLoRAConfig(
        init_seed=42,
        targets=[
            TTLoRATarget(
                module_name_pattern=r".*attn\.c_attn$",
                weight_shape=(2304, 768),
                input_factors=(32, 4, 6),
                output_factors=(48, 6, 8),
                rank=6,
                alpha=16.0,
                variant="contraction",
            )
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--save-adapter-dir", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    config = (
        TTLoRAConfig.from_json_file(args.config_json)
        if args.config_json
        else default_gpt2_c_attn_config()
    )
    model = get_ttlora_model(model, config)
    print_trainable_parameters(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    batch = tokenizer("TT-LoRA adapts selected weights with tensor-train cores.", return_tensors="pt")
    outputs = model(**batch, labels=batch["input_ids"])
    print(f"loss: {outputs.loss.item():.4f}")

    if args.save_adapter_dir:
        save_ttlora_adapters(model, args.save_adapter_dir, config=config)


if __name__ == "__main__":
    main()
