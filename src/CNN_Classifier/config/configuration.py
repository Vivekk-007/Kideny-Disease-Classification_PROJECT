from pathlib import Path
from CNN_Classifier.utils.common import read_yaml, create_directories
from CNN_Classifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
)
from CNN_Classifier import logger


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: str = "config/config.yaml",
        params_filepath: str = "params.yaml",
    ):
        logger.info("Loading configuration files")

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params

        create_directories([config.root_dir])

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=tuple(params.IMAGE_SIZE),
            params_weights=params.WEIGHTS,
            params_include_top=params.INCLUDE_TOP,
            params_classes=params.CLASSES,
            params_learning_rate=params.LEARNING_RATE,
            params_freeze_all=config.freeze_all,
            params_freeze_till=config.freeze_till,
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        params = self.params
        prepare_base_model = self.config.prepare_base_model

        create_directories([training.root_dir])

        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            train_data_dir=Path(training.train_data_dir),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_image_size=tuple(params.IMAGE_SIZE),
            params_is_augmentation=training.augmentation,
            early_stopping=training.early_stopping,
            early_stopping_patience=training.early_stopping_patience,
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        evaluation = self.config.evaluation
        params = self.params

        create_directories([evaluation.root_dir])

        return EvaluationConfig(
            root_dir=Path(evaluation.root_dir),
            model_path=Path(evaluation.model_path),
            test_data_dir=Path(evaluation.test_data_dir),
            params_image_size=tuple(params.IMAGE_SIZE),
            params_batch_size=params.BATCH_SIZE,
        )