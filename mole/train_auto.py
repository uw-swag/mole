import json
import argparse

import torch
from transformers import (
    AutoModel,
    AdamW,
)
from transformers.trainer_utils import get_last_checkpoint

from mole.config import TrainAutoConfig
from mole.common import build_tokenizer, build_datasets, build_trainer


def build_model(model_config, tokenizer):
    model = AutoModel.from_pretrained(
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
        model.embed_tokens.weight.data[tokenizer.pad_token_id] = model.embed_tokens.weight.data[tokenizer.eos_token_id]
    model.config.use_cache = False # Not useful for training

    num_parameters = sum(1 for _ in model.parameters())
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    num_trainale_parameters = len(trainable_parameters)
    print(f"Training {num_trainale_parameters} out of {num_parameters}")
    optimizer = AdamW(trainable_parameters, lr=model_config.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    return model, optimizer


def save_model(trainer):
    output_dir = trainer.args.output_dir
    final_save_dir = output_dir + "/final"

    model = trainer.model
    model.save_pretrained(final_save_dir)

    trainer.tokenizer.save_pretrained(final_save_dir)


def main():
    parser = argparse.ArgumentParser(prog="Full Fine-tune Model Trainer")
    parser.add_argument("base_dir")  
    base_dir = parser.parse_args().base_dir
    with open(base_dir + "/auto_config.json", mode="r") as f:
        config_dict = json.load(f)
    config = TrainAutoConfig.model_validate(config_dict)

    tokenizer = build_tokenizer(config.tokenizer)
    train_dataset, test_datasets = build_datasets(config.datasets, tokenizer)
    model, optimizer = build_model(config.model, tokenizer)
    trainer = build_trainer(base_dir, config.training, tokenizer, train_dataset, test_datasets, model, optimizer)

    import os
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    res = trainer.train(resume_from_checkpoint=get_last_checkpoint(trainer.args.output_dir))
    with open(trainer.args.output_dir + "/train.json", "w") as f:
        json.dump(res , f)

    save_model(trainer)

if __name__ == '__main__':
    main()
