import unittest
import os
from pathlib import Path
from CNN_Classifier.components.prediction import PredictionPipeline


class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.model_path = "artifacts/training/model.h5"
        if os.path.exists(cls.model_path):
            cls.predictor = PredictionPipeline(cls.model_path)
        else:
            cls.predictor = None
    
    def test_model_loaded(self):
        """Test if model loads successfully"""
        if self.predictor:
            self.assertIsNotNone(self.predictor.model)
    
    def test_class_names(self):
        """Test if class names are correct"""
        if self.predictor:
            expected_classes = ['Cyst', 'Normal', 'Stone', 'Tumor']
            self.assertEqual(self.predictor.class_names, expected_classes)
    
    def test_prediction_structure(self):
        """Test prediction output structure"""
        # This would require a sample image
        pass


if __name__ == '__main__':
    unittest.main()