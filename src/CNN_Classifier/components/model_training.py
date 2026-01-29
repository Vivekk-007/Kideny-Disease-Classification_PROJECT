import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from CNN_Classifier.entity.config_entity import TrainingConfig
from CNN_Classifier import logger


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    def get_base_model(self):
        """Load and compile the base model"""
        logger.info("Loading base model from disk")
        
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False
        )
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
        logger.info("Model loaded and compiled successfully")

    def train_valid_generator(self):
        """Create training and validation data generators with improved augmentation"""
        from tensorflow.keras.applications.resnet50 import preprocess_input

        datagenerator_kwargs = dict(
            preprocessing_function=preprocess_input,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size,
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Validation data generator (no augmentation)
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagen.flow_from_directory(
            directory=self.config.train_data_dir,
            subset="validation",
            shuffle=False,
            class_mode="sparse",
            **dataflow_kwargs
        )

        # IMPROVED: Stronger augmentation for training
        if self.config.params_is_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                **datagenerator_kwargs
            )
            logger.info("Training with data augmentation enabled")
        else:
            train_datagen = valid_datagen
            logger.info("Training without data augmentation")

        self.train_generator = train_datagen.flow_from_directory(
            directory=self.config.train_data_dir,
            subset="training",
            shuffle=True,
            class_mode="sparse",
            **dataflow_kwargs
        )

        logger.info(f"Training samples: {self.train_generator.samples}")
        logger.info(f"Validation samples: {self.valid_generator.samples}")
        logger.info(f"Classes: {self.train_generator.class_indices}")

    def train(self):
        """Train the model with improved callbacks and strategies"""
        
        # Compute class weights for imbalanced dataset
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.train_generator.classes),
            y=self.train_generator.classes
        )
        class_weights = dict(enumerate(class_weights))
        
        logger.info(f"Class weights: {class_weights}")

        # IMPROVED: Better callbacks configuration
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.config.trained_model_path).replace('\\', '/'),
                save_best_only=True,
                monitor="val_accuracy",
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                filename=str(self.config.root_dir / 'training_log.csv'),
                append=False
            )
        ]

        # Calculate steps
        steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Validation steps: {validation_steps}")
        logger.info(f"Starting training for {self.config.params_epochs} epochs")

        # Train the model
        history = self.model.fit(
            self.train_generator,
            validation_data=self.valid_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        # Save final model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        
        logger.info("Training completed successfully")
        
        # Log final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        logger.info(f"Final Training Accuracy: {final_train_acc:.4f}")
        logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
        
        return history

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save the trained model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        path_str = str(path.resolve()).replace('\\', '/')
        model.save(path_str)
        logger.info(f"Model saved to: {path_str}")