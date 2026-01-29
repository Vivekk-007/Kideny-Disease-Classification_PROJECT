from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.prepare_base_model import PrepareBaseModel
from CNN_Classifier import logger


STAGE_NAME = "Prepare Base Model Stage"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()

        prepare_base_model = PrepareBaseModel(
            config=prepare_base_model_config
        )

        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()

        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<")


if __name__ == '__main__':
    try:
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
    except Exception as e:
        logger.exception(e)
        raise e