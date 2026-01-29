import tensorflow as tf
from pathlib import Path
from CNN_Classifier.entity.config_entity import PrepareBaseModelConfig
from CNN_Classifier import logger


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.base_model = None
        self.full_model = None

    def get_base_model(self):
        """Load ResNet50 base model"""
        logger.info("Loading ResNet50 base model")

        self.base_model = tf.keras.applications.ResNet50(
            input_shape=(*self.config.params_image_size, 3),
            weights=self.config.params_weights,
            include_top=False
        )
        self.base_model.trainable = False  

        self.save_model(
            path=self.config.base_model_path,
            model=self.base_model
        )

        logger.info(f"Base model saved at: {self.config.base_model_path}")

    @staticmethod
    def _prepare_full_model(
        model: tf.keras.Model,
        classes: int,
        freeze_all: bool,
        freeze_till: int,
        learning_rate: float
    ) -> tf.keras.Model:
        """Prepare full model with custom classification head"""

        logger.info("Preparing full model with custom classification head")

        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
            logger.info("All base model layers frozen")

        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
            for layer in model.layers[-freeze_till:]:
                layer.trainable = True

            logger.info(
                f"Fine-tuning enabled for last {freeze_till} layers of ResNet50"
            )

        # IMPROVED: Better architecture for higher accuracy
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(model.output)
        
        # First dense block
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        # Second dense block
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        # Second dense block
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        # Second dense block
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        # Third dense block
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # Fourth dense block
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        # Fifth dense block
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        
        # Third dense block
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # Output layer
        output = tf.keras.layers.Dense(
            units=classes,
            activation="softmax",
            name='output'
        )(x)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=output
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        logger.info("Full model compiled successfully")
        full_model.summary()
        return full_model

    def update_base_model(self):
        """Update base model with custom head"""
        logger.info("Updating base model with custom head")

        self.full_model = self._prepare_full_model(
            model=self.base_model,
            classes=self.config.params_classes,
            freeze_all=self.config.params_freeze_all,
            freeze_till=self.config.params_freeze_till,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )

        logger.info(
            f"Updated base model saved at: {self.config.updated_base_model_path}"
        )

    @staticmethod
    def save_model(path, model: tf.keras.Model):
        """Save model with proper Windows path handling"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        abs_path = path.resolve()
        path_str = str(abs_path).replace('\\', '/')
        
        try:
            model.save(path_str)
            logger.info(f"Model successfully saved to: {path_str}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            keras_path = str(abs_path).replace('.h5', '.keras').replace('\\', '/')
            logger.info(f"Attempting to save with .keras format: {keras_path}")
            model.save(keras_path)