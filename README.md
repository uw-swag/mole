# MoLE

MoLE (Mix-of-Language-Experts) is a novel LLM model architecture for multilingual programming. Quoting the abstract of [our LLM4Code'25 paper](https://pengyunie.github.io/p/ZongETAL25MoLE.pdf):

> Large language models (LLMs) have demonstrated impressive capabilities in aiding developers with tasks like code comprehension, generation, and translation. Supporting multilingual programming---i.e., coding tasks across multiple programming languages---typically requires either (1) finetuning a single LLM across all programming languages, which is cost-efficient but sacrifices language-specific specialization and performance, or (2) finetuning separate LLMs for each programming language, which allows for specialization but is computationally expensive and storage-intensive due to the duplication of parameters.
> This paper introduces MoLE (Mix-of-Language-Experts), a novel architecture that balances efficiency and specialization for multilingual programming. MoLE is composed of a base model, a shared LoRA (low-rank adaptation) module, and a collection of language-specific LoRA modules. These modules are jointly optimized during the finetuning process, enabling effective knowledge sharing and specialization across programming languages. During inference, MoLE automatically routes to the language-specific LoRA module corresponding to the programming language of the code token being generated. Our experiments demonstrate that MoLE achieves greater parameter efficiency compared to training separate language-specific LoRAs, while outperforming a single shared LLM finetuned for all programming languages in terms of accuracy.


This repository contains the code for MoLE.

## Table of Contents

- [Usage](#usage)
- [Citation](#citation)

## Usage

### Prerequisites

- Linux operation system
- Python 3.9+
- [CUDA 12.3+](https://developer.nvidia.com/cuda-toolkit)

### Setup Python environment

The dependencies of MoLE are listed in `requirements.txt`. You can install them by running (recommended in a virtual environment, e.g., `virtualenv`):
```
pip install -r requirements.txt
```

`flash-attn` needs to be installed separately. You can follow the instructions at [Dao-AILab/flash-attn](https://github.com/Dao-AILab/flash-attention).

### Preparing dataset

Run `python3 ./scripts/filter_glaive_dataset.py` to download and filter the dataset from HuggingFace. The result is stored at `./datasets/filtered_glaive`.

### Finetuning

Create a diretory `{dir}` to contain the artifacts.

For
* full baseline, copy `./mole/auto_config.json` into `{dir}`.
* LoRA baseline, copy `./mole/lora_config.json` into `{dir}`.
* MoLE, copy `./mole/moe_config.json` into `{dir}`.

Set `{auto|lora|moe}_config.datasets.path_on_disk` to the full path of the processed dataset then start training with `python3 ./mole/train_{auto|lora|moe}.py <dir>`.

### Inference

The tokenizer and finetuned model are saved at `{dir}/final`. Load them using `transformers.AutoTokenizer` and `transformers.AutoModel`.


## Citation

If you use MoLE in your work, please cite the following paper:

```
@inproceedings{ZongETAL25MoLE,
    title={Mix-of-Language-Experts Architecture for Multilingual Programming},
    author={Zong, Yifan and Deng, Yuntian and Nie, Pengyu},
    booktitle={International Workshop on Large Language Models for Code},
    year={2025},
}
```
