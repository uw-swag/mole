from typing import List, Dict
from enum import Enum
from pydantic import BaseModel


class TokenizerConfig(BaseModel):
    path: str


class DatasetsConfig(BaseModel):
    path: str | None = None
    path_on_disk: str | None = None
    truncation_length: int
    mask_problem_labels: bool = True
    shuffle_seed: int = 42
    subset: str | None = None


class AutoModelConfig(BaseModel):
    path: str
    learning_rate: float


class LoraModelConfig(BaseModel):
    path: str
    lora_path: str | None
    rank: int
    alpha: int
    learning_rate: float
    modules_to_save: List[str] = []


class MoEInitMethod(Enum):
    SHARED_LAST = "shared_last"
    SHARED_FIRST  = "shared_first"
    STD = "std"
    STD_ZERO = "std_zero"


class MoEModelConfig(BaseModel):
    moe_path: str | None = None
    base_path: str | None = None
    expert_rank: int
    shared_expert_rank: int
    expert_alpha: int | None = None
    shared_expert_alpha: int | None = None
    other_expert_alpha: int | None = None
    init_method: MoEInitMethod = MoEInitMethod.SHARED_LAST
    learning_rate: float
    group_learning_rates: Dict[str, float] = {}


class TrainingConfig(BaseModel):
    epochs: int
    per_device_train_batch_size: int
    per_device_test_batch_size: int
    gradient_accumulation_steps: int
    num_save_evals: int = 25


class TrainAutoConfig(BaseModel):
    tokenizer: TokenizerConfig
    datasets: DatasetsConfig
    model: AutoModelConfig
    training: TrainingConfig


class TrainLoraConfig(BaseModel):
    tokenizer: TokenizerConfig
    datasets: DatasetsConfig
    model: LoraModelConfig
    training: TrainingConfig


class ExpertsType(Enum):
    OTHER = "other" # Use another expert ("other)") for non-code
    SHARE_OTHER = "share_other" # Use the shared and another expert ("shared" and "expert") for non-code
    NO_CODE = "no_code" # Don't use individual experts (only use "shared") for code
    NO_SHARE = "no_share" # Don't use shared expert
    NO_OTHER = "no_other" # Don't use other


class ExpertsConfig(BaseModel):
    expert_type: ExpertsType
    expert_names: List[str]


class TrainMoEConfig(BaseModel):
    experts: ExpertsConfig
    tokenizer: TokenizerConfig
    datasets: DatasetsConfig
    model: MoEModelConfig
    training: TrainingConfig
