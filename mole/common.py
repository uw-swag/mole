from string import Template
from functools import partial
from itertools import repeat
import copy
import warnings
import json

from datasets import load_from_disk, load_dataset
from transformers import (
    LlamaTokenizerFast,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
    default_data_collator,
)

from mole.config import ExpertsType

PREAMBLE = "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions."
INSTRUCTION = "\n\n@@ Instruction\n"
RESPONSE = "\n\n@@ Response\n"

CHAT_TEMPLATE = Template(
    "{{bos_token}}{{'$PREAMBLE'}}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n        {{ raise_exception('System messages are not allowed in this template.') }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'$INSTRUCTION' + message['content']}}\n        {%- else %}\n{{'$RESPONSE' + message['content'] + eos_token}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{'$RESPONSE'}}{%- endif %}\n"
).substitute(PREAMBLE=PREAMBLE, INSTRUCTION=INSTRUCTION, RESPONSE=RESPONSE)


def build_tokenizer(tokenizer_config):
    tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_config.path, trust_remote_code=True)

    if tokenizer_config.path == "deepseek-ai/deepseek-coder-1.3b-base":
        tokenizer.chat_template = CHAT_TEMPLATE
        tokenizer.pad_token = "<pad>" # Use a different pad token
    return tokenizer


def build_datasets(datasets_config, tokenizer, experts_config=None):
    if datasets_config.path:
        datasets = load_dataset(datasets_config.path)
        train_split = datasets["train"]
        test_split = datasets["test"]
        assert datasets_config.path_on_disk is None
    else:
        train_split = load_from_disk(f"{datasets_config.path_on_disk}_train")
        test_split = load_from_disk(f"{datasets_config.path_on_disk}_test")
        assert datasets_config.path is None

    # Find all unique langs in the dataset
    all_unique_langs = set(lang for langs in train_split["langs"] + test_split["langs"] for lang in langs)
    lang_to_id = dict.fromkeys(all_unique_langs, None)
    for i, expert in enumerate(lang_to_id):
        lang_to_id[expert] = i
    print(f"Languages in dataset: {all_unique_langs}")

    if experts_config is not None:
        required_keys = []
        if experts_config.expert_type == ExpertsType.NO_CODE:
            required_keys = ["shared", "other"]
        elif experts_config.expert_type == ExpertsType.OTHER or experts_config.expert_type == ExpertsType.SHARE_OTHER:
            required_keys = ["shared", "other"]
            required_keys.extend(lang_to_id)
        elif experts_config.expert_type == ExpertsType.NO_SHARE:
            required_keys = ["other"]
            required_keys.extend(lang_to_id)
        elif experts_config.expert_type == ExpertsType.NO_OTHER:
            required_keys = ["shared"]
            required_keys.extend(lang_to_id)
        else:
            assert False
        assert all(key in experts_config.expert_names for key in required_keys)
        expert_to_id = {key: idx for idx, key in enumerate(experts_config.expert_names)}

    # Number of fixed chat_template tokens before problem and instruction respectively
    _PRE_PROBLEM_LEN = len(tokenizer.bos_token) + len(PREAMBLE) + len(INSTRUCTION)
    _PRE_SOLUTION_LEN = _PRE_PROBLEM_LEN + len(RESPONSE)

    def _build_sample(examples, mask_problem_labels=True):
        chats = [
            [
                {"role": "user", "content": p},
                {"role": "model", "content": s},
            ] for p, s in zip(examples["problem"], examples["solution"])
        ]
        prompts = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=False)
        batch_encoding = tokenizer(prompts, add_special_tokens=False, truncation=True, max_length=datasets_config.truncation_length)

        outputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        expert_indices = []
        for i, (problem, input_ids, attention_mask) in enumerate(
            zip(examples["problem"], batch_encoding.input_ids, batch_encoding.attention_mask)
        ):
            solution_token_start = batch_encoding.char_to_token(i, _PRE_SOLUTION_LEN + len(problem))
            if solution_token_start is None:
                warnings.warn(f'"{prompts[i]}" is dropped because solution is truncated')
                continue

            outputs["input_ids"].append(input_ids)
            outputs["attention_mask"].append(attention_mask)

            labels = copy.deepcopy(input_ids)
            if mask_problem_labels:
                # Replace instruction tokens with -100
                labels[:solution_token_start] = repeat(-100, times=solution_token_start)
            else:
                # Replace template tokens with -100
                instruction_token_start = batch_encoding.char_to_token(i, _PRE_PROBLEM_LEN)
                instruction_token_end = batch_encoding.char_to_token(i, _PRE_PROBLEM_LEN + len(problem))
                labels[:instruction_token_start] = repeat(-100, times=instruction_token_start)
                labels[instruction_token_end:solution_token_start] = repeat(
                    -100, times=(solution_token_start - instruction_token_end)
                )
            outputs["labels"].append(labels)

            if experts_config is not None:
                problem_code_blocks = json.loads(examples["problem_code_blocks"][i])
                solution_code_blocks = json.loads(examples["solution_code_blocks"][i])

                # Create expert_indices
                if experts_config.expert_type == ExpertsType.OTHER:
                    code_indices = [None, expert_to_id["shared"]]
                    non_code_indices = [expert_to_id["other"], -1]
                elif experts_config.expert_type == ExpertsType.SHARE_OTHER:
                    code_indices = [None, expert_to_id["shared"]]
                    non_code_indices = [expert_to_id["other"], expert_to_id["shared"]]
                elif experts_config.expert_type == ExpertsType.NO_CODE:
                    code_indices = [expert_to_id["shared"]]
                    non_code_indices = [expert_to_id["other"]]
                elif experts_config.expert_type == ExpertsType.NO_SHARE:
                    code_indices = [None]
                    non_code_indices = [expert_to_id["other"]]
                elif experts_config.expert_type == ExpertsType.NO_OTHER:
                    langs = {}
                    for (start, end), lang in problem_code_blocks:
                        langs[lang] = langs.get(lang, 0) + (end - start)
                    for (start, end), lang in solution_code_blocks:
                        langs[lang] = langs.get(lang, 0) + (end - start)
                    principal_lang = max(langs, key=langs.get)
                    code_indices = non_code_indices = [expert_to_id[principal_lang], expert_to_id["shared"]]
                else:
                    assert False

                token_indices = [non_code_indices] * len(input_ids)

                # Code blocks in problem
                for (start, end), lang in problem_code_blocks:
                    code_token_start = batch_encoding.char_to_token(i, _PRE_PROBLEM_LEN + start)
                    code_token_end = batch_encoding.char_to_token(i, _PRE_PROBLEM_LEN + end)

                    _code_indices = copy.copy(code_indices)
                    if experts_config.expert_type != ExpertsType.NO_CODE:
                        _code_indices[0] = expert_to_id[lang]

                    token_indices[code_token_start:code_token_end] = repeat(
                        _code_indices, code_token_end - code_token_start
                    )

                # Code blocks in solution
                for (start, end), lang in solution_code_blocks:
                    code_token_start = batch_encoding.char_to_token(i, _PRE_SOLUTION_LEN + len(problem) + start)
                    if code_token_start is None: # Skip if entire block is truncated
                        break
                    code_token_end = batch_encoding.char_to_token(i, _PRE_SOLUTION_LEN + len(problem) + end)

                    _code_indices = copy.copy(code_indices)
                    if experts_config.expert_type != ExpertsType.NO_CODE:
                        _code_indices[0] = expert_to_id[lang]

                    if code_token_end is None: # Block end is truncated
                        token_indices[code_token_start:] = repeat(
                            _code_indices, len(token_indices) - code_token_start
                        )
                        break
                    else:
                        token_indices[code_token_start:code_token_end] = repeat(
                            _code_indices, code_token_end - code_token_start
                        )

                expert_indices.append(token_indices)
                outputs["expert_indices"] = expert_indices

        return outputs

    train_dataset = train_split.shuffle(seed=datasets_config.shuffle_seed)

    test_datasets = {}
    if datasets_config.subset is None:
        for group in test_split.features["group"].names:
            test_datasets[group] = test_split.filter(
                lambda x: x["group"] == test_split.features["group"].str2int(group)
            ).map(_build_sample, batched=True, remove_columns=test_split.column_names)
    else:
        group = datasets_config.subset
        assert group in test_split.features["group"].names
        train_dataset = train_dataset.filter(lambda x: x["group"] == test_split.features["group"].str2int(group))
        test_datasets[group] = test_split.filter(
            lambda x: x["group"] == test_split.features["group"].str2int(group)
        ).map(_build_sample, batched=True, remove_columns=test_split.column_names)

    train_dataset = train_dataset.map(
        partial(_build_sample, mask_problem_labels=datasets_config.mask_problem_labels), batched=True, remove_columns=test_split.column_names
    )

    return train_dataset, test_datasets


def build_expert_indices_collator(tokenizer):
    def collate(features: list[dict]):
        max_len = max(len(example["input_ids"]) for example in features)

        for example in features:
            len_diff = (max_len - len(example["input_ids"]))
            example["input_ids"] += [tokenizer.pad_token_id] * len_diff
            example["labels"] += [-100] * len_diff
            example["attention_mask"] += [0] * len_diff
            if "expert_indices" in example:
                example["expert_indices"] += [[-1] * len(example["expert_indices"][0])] * len_diff

        return default_data_collator(features)
    return collate


def build_trainer(output_dir, training_config, tokenizer, train_dataset, test_datasets, model, optimizer, compile=True): 
    total_samples = len(train_dataset) * training_config.epochs
    samples_per_step = training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps
    max_steps = total_samples // samples_per_step
    num_warmup_steps = round(max_steps * 0.05)
    num_training_steps = round(max_steps * 1.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    save_eval_ratio = 1. / training_config.num_save_evals
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_test_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        max_grad_norm=None,
        group_by_length=False,
        save_strategy="steps",
        save_steps=save_eval_ratio,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=save_eval_ratio,
        prediction_loss_only=True,
        fp16_full_eval=True,
        bf16=True,
        tf32=True,
        torch_compile=compile,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler),
        train_dataset=train_dataset,
        eval_dataset=test_datasets,
        data_collator=build_expert_indices_collator(tokenizer),
    )

    return trainer
