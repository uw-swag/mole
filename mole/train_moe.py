import json
import argparse

import torch
from transformers import (
    AdamW,
)
from transformers.trainer_utils import get_last_checkpoint

from mole.moe_model import MoEForCausalLM
from mole.config import TrainMoEConfig, MoEModelConfig, MoEInitMethod, ExpertsConfig, ExpertsType
from mole.common import build_tokenizer, build_datasets, build_trainer


def build_model(model_config: MoEModelConfig, experts_config: ExpertsConfig, tokenizer):
    if model_config.moe_path is not None:
        assert model_config.base_path is None
        model = MoEForCausalLM.from_pretrained(
            model_config.moe_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
    else:
        code_rank = model_config.expert_rank
        code_alpha = model_config.expert_alpha
        shared_rank = model_config.shared_expert_rank
        shared_alpha = model_config.shared_expert_alpha

        if model_config.init_method == MoEInitMethod.SHARED_LAST:
            code_init_method = "svd_0"
            shared_init_method = f"svd_{code_rank}"
        elif model_config.init_method == MoEInitMethod.SHARED_FIRST:
            code_init_method = f"svd_{shared_rank}"
            shared_init_method = "svd_0"
        elif model_config.init_method == MoEInitMethod.STD_ZERO:
            code_init_method = "std_zero"
            shared_init_method = "std_zero"
        else:
            assert False

        other_alpha = model_config.other_expert_alpha
        if experts_config.expert_type == ExpertsType.SHARE_OTHER:
            other_rank = code_rank
            other_init_method = code_init_method
        else:
            other_rank = code_rank + shared_rank
            other_init_method = min(code_init_method, shared_init_method)

        num_experts = len(experts_config.expert_names)
        expert_rank = [None] * num_experts
        expert_alpha = [None] * num_experts
        expert_init_method = [None] * num_experts
        for idx, expert_name in enumerate(experts_config.expert_names):
            if expert_name == "shared":
                expert_rank[idx] = shared_rank
                expert_alpha[idx] = shared_alpha
                expert_init_method[idx] = shared_init_method
            elif expert_name == "other":
                expert_rank[idx] = other_rank
                expert_alpha[idx] = other_alpha
                expert_init_method[idx] = other_init_method
            else:
                expert_rank[idx] = code_rank
                expert_alpha[idx] = code_alpha
                expert_init_method[idx] = code_init_method

        model = MoEForCausalLM.from_pretrained_llama(
            model_config.base_path,
            num_experts=num_experts,
            expert_rank=expert_rank,
            expert_alpha=expert_alpha,
            expert_init_method=expert_init_method,
            default_experts=[],
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
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

    param_groups = {group: {"params": [], "lr": lr} for group, lr in model_config.group_learning_rates.items()}
    other_expert_params = []
    for num_params, (name, param) in enumerate(model.named_parameters(), start=1):
        param.requires_grad = True
        for group, param_group in param_groups.items():
            params = param_group["params"]
            if group in name:
                params.append(param)
                break
        else:
            if ".experts." in name:
                other_expert_params.append(param)
            else:
                param.requires_grad = False
        continue
    param_groups["other_experts"] = {"params": other_expert_params, "lr": model_config.learning_rate}

    print(f"Total parameters: {num_params}")
    param_group_log = {group: len(param_group["params"]) for group, param_group in param_groups.items()}
    print(f"Training: {param_group_log}")

    optimizer = AdamW(param_groups.values(), betas=(0.9, 0.95), weight_decay=0.0)
    return model, optimizer


def save_model(trainer):
    final_save_dir = trainer.args.output_dir + "/final"
    trainer.model.config.use_cache = True
    trainer.model.config.expert_init_method = "none"
    trainer.save_model(final_save_dir)
    trainer.tokenizer.save_pretrained(final_save_dir)


def update_code_indices(test_datasets, experts_config: ExpertsConfig, lang):
    expert_names = experts_config.expert_names
    lang_id = expert_names.index(lang)
    shared_id = other_id = -100 # init with invalid id
    if (experts_config.expert_type != ExpertsType.NO_SHARE):
        shared_id = expert_names.index("shared")
    if (experts_config.expert_type != ExpertsType.NO_OTHER):
        other_id = expert_names.index("other")

    def _update_code_indices(examples):
        for expert_indices in examples["expert_indices"]:
            for token_indices in expert_indices:
                for i in range(len(token_indices)):
                    if token_indices[i] != shared_id and token_indices[i] != other_id:
                        token_indices[i] = lang_id
        return examples

    for lang in test_datasets:
        test_datasets[lang] = test_datasets[lang].map(
            _update_code_indices, batched=True
        )

def main():
    parser = argparse.ArgumentParser(prog="MoE Model Trainer")
    parser.add_argument("base_dir")  
    base_dir = parser.parse_args().base_dir
    with open(base_dir + "/moe_config.json", mode="r") as f:
        config_dict = json.load(f)
    config = TrainMoEConfig.model_validate(config_dict)

    tokenizer = build_tokenizer(config.tokenizer)
    train_dataset, test_datasets = build_datasets(config.datasets, tokenizer, config.experts)
    model, optimizer = build_model(config.model, config.experts, tokenizer)
    trainer = build_trainer(base_dir, config.training, tokenizer, train_dataset, test_datasets, model, optimizer)

    import os
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    res = trainer.train(resume_from_checkpoint=get_last_checkpoint(trainer.args.output_dir))
    with open(trainer.args.output_dir + "/train.json", "w") as f:
        json.dump(res , f)

    if config.experts.expert_type == ExpertsType.NO_CODE:
        res = trainer.evaluate(test_datasets)
    else:
        res = {}
        for lang in test_datasets:
            if lang in config.experts.expert_names:
                update_code_indices(test_datasets, config.experts, lang)
                res[lang] = trainer.evaluate(test_datasets)

    with open(trainer.args.output_dir + "/eval.json", "w") as f:
        json.dump(res , f)

    save_model(trainer)

if __name__ == '__main__':
    main()
