from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.model_training import Training
from CNN_Classifier import logger

# Update config to use smaller batch and fewer epochs for testing
config = ConfigurationManager()
training_config = config.get_training_config()

# Override for quick testing
training_config = training_config._replace(
    params_batch_size=8,
    params_epochs=5
)

training = Training(config=training_config)
training.get_base_model()
training.train_valid_generator()

# Check class distribution
print("\n" + "="*50)
print("CLASS DISTRIBUTION:")
print("="*50)
for class_name, class_idx in training.train_generator.class_indices.items():
    count = sum(training.train_generator.classes == class_idx)
    print(f"{class_name}: {count} samples")
print("="*50 + "\n")

training.train()