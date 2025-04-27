import json
import argparse

import torch
from transformers import (
    LlamaForCausalLM,
    AdamW,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, PeftModel, get_peft_model

from mole.config import TrainLoraConfig
from mole.common import build_tokenizer, build_datasets, build_trainer


def build_model(model_config, tokenizer):
    model = LlamaForCausalLM.from_pretrained(
        model_config.path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda"
    )

    if model.config.eos_token_id != tokenizer.eos_token_id or model.config.pad_token_id != tokenizer.pad_token_id:
        # Initialize pad token embed to be the same as eos token embed (as in pretraining)
        print(f"Now using EOS token {tokenizer.eos_token}: {tokenizer.eos_token_id}")
        print(f"Now using PAD token {tokenizer.pad_token}: {tokenizer.pad_token_id}")
        print("Pad token's embeddings are copied from EOS token's.")
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.model.embed_tokens.weight.data[tokenizer.pad_token_id] = model.model.embed_tokens.weight.data[tokenizer.eos_token_id]
    model.config.use_cache = False # Not useful for training

    if model_config.lora_path is not None:
        print("Loading existing LoRA")
        model = PeftModel.from_pretrained(model, model_config.lora_path, is_trainable=True)
    else:
        print("Creating new LoRA")
        lora_config = LoraConfig(
            r=model_config.rank, lora_alpha=model_config.alpha,
            bias="none", use_rslora=True, init_lora_weights="pissa",
            target_modules=["gate_proj", "up_proj", "down_proj"],
            modules_to_save=model_config.modules_to_save,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=model_config.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    return model, optimizer


def save_model(trainer):
    output_dir = trainer.args.output_dir
    base_model_save_dir = output_dir + "/base"
    adapter_model_save_dir = output_dir + "/adapter"
    final_save_dir = output_dir + "/final"

    model = trainer.model
    model.config.use_cache = True
    for config in model.peft_config.values(): # Trick lora into not doing svd again
        config.init_lora_weights = False
    model.save_pretrained(adapter_model_save_dir)
    model = model.unload()
    model.save_pretrained(base_model_save_dir)

    model = LlamaForCausalLM.from_pretrained(
        base_model_save_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, adapter_model_save_dir)
    model = model.merge_and_unload()
    model.save_pretrained(final_save_dir)

    trainer.tokenizer.save_pretrained(final_save_dir)


def main():
    parser = argparse.ArgumentParser(prog="LoRA Model Trainer")
    parser.add_argument("base_dir")  
    base_dir = parser.parse_args().base_dir
    with open(base_dir + "/lora_config.json", mode="r") as f:
        config_dict = json.load(f)
    config = TrainLoraConfig.model_validate(config_dict)

    tokenizer = build_tokenizer(config.tokenizer)    
    model, optimizer = build_model(config.model, tokenizer)
    train_dataset, test_datasets = build_datasets(config.datasets, tokenizer)
    trainer = build_trainer(base_dir, config.training, tokenizer, train_dataset, test_datasets, model, optimizer, compile=False)

    import os
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    res = trainer.train(resume_from_checkpoint=get_last_checkpoint(trainer.args.output_dir))
    with open(trainer.args.output_dir + "/train.json", "w") as f:
        json.dump(res , f)

    res = trainer.evaluate()
    with open(trainer.args.output_dir + "/eval.json", "w") as f:
        json.dump(res , f) 

    save_model(trainer)

if __name__ == '__main__':
    main()
