import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_recall_fscore_support
)
from CNN_Classifier.entity.config_entity import EvaluationConfig
from CNN_Classifier import logger


class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.test_generator = None
        
    def load_model(self):
        """Load trained model"""
        logger.info(f"Loading model from {self.config.model_path}")
        self.model = tf.keras.models.load_model(self.config.model_path)
        logger.info("Model loaded successfully")
        
    def prepare_test_data(self):
        """Prepare test data generator"""
        from tensorflow.keras.applications.resnet50 import preprocess_input
        
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            directory=self.config.test_data_dir,
            target_size=self.config.params_image_size,
            batch_size=self.config.params_batch_size,
            class_mode='sparse',
            shuffle=False
        )
        
        logger.info(f"Test samples: {self.test_generator.samples}")
        logger.info(f"Classes: {self.test_generator.class_indices}")
        
    def evaluate(self):
        """Evaluate model on test set"""
        logger.info("Evaluating model on test set...")
        
        # Get predictions
        predictions = self.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=1)
        
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        class_names = list(self.test_generator.class_indices.keys())
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_true, y_pred, target_names=class_names))
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Save results
        results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classification_report': report
        }
        
        results_path = self.config.root_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {results_path}")
        
        return results, y_true, y_pred, predictions
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        class_names = list(self.test_generator.class_indices.keys())
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = self.config.root_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
        plt.close()
        
    def plot_roc_curves(self, y_true, predictions):
        """Plot ROC curves for each class"""
        from sklearn.preprocessing import label_binarize
        
        class_names = list(self.test_generator.class_indices.keys())
        n_classes = len(class_names)
        
        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
            auc_score = roc_auc_score(y_true_bin[:, i], predictions[:, i])
            plt.plot(
                fpr, tpr, 
                color=color, 
                lw=2,
                label=f'{class_name} (AUC = {auc_score:.3f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for All Classes', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.config.root_dir / 'roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
        plt.close()
        
    def save_sample_predictions(self, num_samples=20):
        """Save sample predictions with images"""
        import random
        
        # Get random samples
        total_samples = min(num_samples, self.test_generator.samples)
        indices = random.sample(range(self.test_generator.samples), total_samples)
        
        rows = 4
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
        axes = axes.ravel()
        
        class_names = list(self.test_generator.class_indices.keys())
        
        for idx, ax in enumerate(axes):
            if idx < len(indices):
                # Get image and prediction
                img_path = self.test_generator.filepaths[indices[idx]]
                img = tf.keras.preprocessing.image.load_img(
                    img_path, 
                    target_size=self.config.params_image_size
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                from tensorflow.keras.applications.resnet50 import preprocess_input
                img_array = preprocess_input(img_array)
                
                prediction = self.model.predict(img_array, verbose=0)
                pred_class = np.argmax(prediction)
                confidence = np.max(prediction)
                
                true_class = self.test_generator.classes[indices[idx]]
                
                ax.imshow(img)
                ax.axis('off')
                
                color = 'green' if pred_class == true_class else 'red'
                title = (
                    f'True: {class_names[true_class]}\n'
                    f'Pred: {class_names[pred_class]}\n'
                    f'Confidence: {confidence:.2%}'
                )
                ax.set_title(title, color=color, fontsize=9, fontweight='bold')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        save_path = self.config.root_dir / 'sample_predictions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sample predictions saved to {save_path}")
        plt.close()
        
    def plot_class_distribution(self, y_true, y_pred):
        """Plot class distribution comparison"""
        class_names = list(self.test_generator.class_indices.keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # True distribution
        unique, counts = np.unique(y_true, return_counts=True)
        ax1.bar([class_names[i] for i in unique], counts, color='skyblue', edgecolor='black')
        ax1.set_title('True Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # Predicted distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        ax2.bar([class_names[i] for i in unique_pred], counts_pred, color='lightcoral', edgecolor='black')
        ax2.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.config.root_dir / 'class_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
        plt.close()