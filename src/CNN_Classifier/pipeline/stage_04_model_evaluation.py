from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.model_evaluation import ModelEvaluation
from CNN_Classifier import logger

STAGE_NAME = "Model Evaluation"


class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluator = ModelEvaluation(config=evaluation_config)
        
        evaluator.load_model()
        evaluator.prepare_test_data()
        results, y_true, y_pred, predictions = evaluator.evaluate()
        evaluator.plot_confusion_matrix(y_true, y_pred)
        evaluator.plot_roc_curves(y_true, predictions)
        evaluator.plot_class_distribution(y_true, y_pred)
        evaluator.save_sample_predictions()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e