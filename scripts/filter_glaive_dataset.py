# %%
import re
import json
from datasets import load_dataset

save_path = "../datasets/filtered_glaive"
langs = ["python", "cpp", "javascript", "java", "rust", "c", "csharp", "go"]


# %%
dataset = load_dataset("glaiveai/glaive-code-assistant-v3", split="train")

# %%
split_pattern = re.compile(r"[ ,;(){}\[\]\+\-\*/=<>!&|\s]")
def split_code(txt):
    return filter(lambda x: x != "", split_pattern.split(txt))
def estimate_num_tokens(txt):
    return sum(1 for _ in split_code(txt))

# %%
code_block_pattern = re.compile(r"```([a-zA-Z0-9_]+)(.+?```)", flags=re.DOTALL|re.IGNORECASE)
def parse_code_langs(examples):
    result = {
        "problem": [],
        "solution": [],
        "langs": [],
        "problem_code_blocks": [],
        "solution_code_blocks": [],
    }

    for question, answer in zip(examples["question"], examples["answer"]):
        lang_lens = {}
        problem_code_blocks = []
        # Find code blocks in question
        for match in code_block_pattern.finditer(question):
            lang = match.group(1).lower()
            if lang not in langs:
                break
            start, end = match.start(2), match.end(2)
            assert end - start > 0, question
            problem_code_blocks.append(((start, end), lang))

            # Estimate number of tokens in code block
            l = estimate_num_tokens(question[start:end-3]) # Ignore final ```
            if lang in lang_lens:
                lang_lens[lang] += l
            else:
                lang_lens[lang] = l
        else:
            # Find code blocks in answer
            solution_code_blocks = []
            for match in code_block_pattern.finditer(answer):
                lang = match.group(1).lower()
                if lang not in langs:
                    break
                start, end = match.start(2), match.end(2)
                assert end - start > 0, answer
                solution_code_blocks.append(((start, end), lang))

                # Count number of words in code block
                l = estimate_num_tokens(answer[start:end-3]) # Ignore final ```
                if lang in lang_lens:
                    lang_lens[lang] += l
                else:
                    lang_lens[lang] = l
            else:
                # Skip samples with <33% code
                if sum(lang_lens.values()) * 3 >= estimate_num_tokens(question) + estimate_num_tokens(answer):
                    result["problem"].append(question)
                    result["solution"].append(answer)
                    # Sort code blocks by lengths
                    result["langs"].append(
                        sorted(lang_lens, key=lambda lang: lang_lens[lang], reverse=True)
                    )
                    # Code blocks are non-overlapping and sorted by end
                    result["problem_code_blocks"].append(json.dumps(problem_code_blocks))
                    result["solution_code_blocks"].append(json.dumps(solution_code_blocks))
    return result

# %%
dataset_w_langs = dataset.map(parse_code_langs, batched=True, remove_columns=dataset.column_names)
all_occurences = [lang for langs in dataset_w_langs["langs"] for lang in langs]
all_langs = set(all_occurences)
lang_to_occurences = [(lang, all_occurences.count(lang)) for lang in all_langs]
lang_to_occurences.sort(key=lambda x: x[1], reverse=True)
print(lang_to_occurences)

# %%
dataset_w_langs = dataset_w_langs.add_column(name="group", column=[langs[0] for langs in dataset_w_langs["langs"]]) # Principal lang of a sample
dataset_w_langs = dataset_w_langs.class_encode_column("group")
datasets = dataset_w_langs.train_test_split(test_size=0.05, seed=42, stratify_by_column="group")

# %%
datasets["train"].save_to_disk(f"{save_path}_train")
datasets["test"].save_to_disk(f"{save_path}_test")
