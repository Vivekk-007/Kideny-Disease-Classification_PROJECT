from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: tuple
    params_learning_rate: float
    params_classes: int
    params_weights: str
    params_include_top: bool  
    params_freeze_all: bool
    params_freeze_till: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    train_data_dir: Path
    params_epochs: int
    params_batch_size: int
    params_learning_rate: float   
    params_image_size: tuple
    params_is_augmentation: bool
    early_stopping: bool
    early_stopping_patience: int


@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    model_path: Path
    test_data_dir: Path
    params_image_size: tuple
    params_batch_size: int