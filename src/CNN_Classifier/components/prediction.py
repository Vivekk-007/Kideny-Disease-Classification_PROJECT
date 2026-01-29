import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
from CNN_Classifier import logger


class PredictionPipeline:
    def __init__(self, model_path: str):
        """Initialize prediction pipeline"""
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
        logger.info(f"Model loaded from {model_path}")
        
    def predict(self, image_path: str):
        """Predict single image"""
        from tensorflow.keras.applications.resnet50 import preprocess_input
        
        try:
            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(
                image_path,
                target_size=(224, 224)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            result = {
                'class': self.class_names[predicted_class_idx],
                'confidence': float(confidence),
                'all_probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.class_names, predictions[0])
                }
            }
            
            logger.info(f"Prediction: {result['class']} ({result['confidence']:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise e