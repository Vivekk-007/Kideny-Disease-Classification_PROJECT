# 🏥 Kidney Disease Classification System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI-powered deep learning system for automated kidney disease classification from CT scans. Achieves 95%+ validation accuracy using ResNet50 architecture with custom classification head.

## 🎯 Features

- **High Accuracy**: 95%+ validation accuracy
- **4 Disease Classes**: Normal, Cyst, Stone, Tumor
- **Real-time Predictions**: Fast inference with confidence scores
- **Web Interface**: User-friendly Flask application
- **REST API**: Easy integration with other systems
- **Docker Support**: Containerized deployment
- **Comprehensive Evaluation**: Detailed metrics and visualizations

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 94.49% |
| Validation Accuracy | 100% |
| Model Architecture | ResNet50 + Custom Head |
| Input Size | 224x224x3 |
| Classes | 4 |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip
- Virtual environment (recommended)

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/kidney-disease-classifier.git
cd kidney-disease-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training the Model
```bash
# Run complete training pipeline
python main.py
```

This will:
1. Download dataset from Google Drive
2. Prepare ResNet50 base model
3. Train the model with augmentation
4. Evaluate on test set
5. Generate evaluation metrics and visualizations

### Running Web Application
```bash
# Start Flask server
python app.py

# Access at: http://localhost:8080
```

## 🐳 Docker Deployment
```bash
# Build image
docker build -t kidney-classifier .

# Run container
docker run -p 8080:8080 kidney-classifier

# Or use docker-compose
docker-compose up
```

## 📁 Project Structure
```
Kideny-Disease-Classification_PROJECT/
├── config/
│   └── config.yaml              # Configuration file
├── params.yaml                  # Model parameters
├── src/
│   └── CNN_Classifier/
│       ├── components/          # Core components
│       ├── pipeline/            # Pipeline stages
│       ├── config/              # Configuration management
│       ├── entity/              # Data entities
│       └── utils/               # Utility functions
├── templates/
│   └── index.html               # Web interface
├── artifacts/                   # Generated artifacts
│   ├── data_ingestion/          # Downloaded data
│   ├── prepare_base_model/      # Base models
│   ├── training/                # Trained model
│   └── evaluation/              # Evaluation results
├── app.py                       # Flask application
├── main.py                      # Training pipeline
├── requirements.txt             # Dependencies
├── Dockerfile                   # Docker configuration
└── README.md                    # Documentation
```

## 🔧 API Usage

### Health Check
```bash
curl http://localhost:8080/health
```

### Make Prediction
```python
import requests

url = "http://localhost:8080/predict"
files = {'file': open('ct_scan.jpg', 'rb')}
response = requests.post(url, files=files)
result = response.json()

print(f"Class: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Response Format
```json
{
  "class": "Normal",
  "confidence": 0.9876,
  "all_probabilities": {
    "Normal": 0.9876,
    "Cyst": 0.0089,
    "Stone": 0.0025,
    "Tumor": 0.0010
  }
}
```

## 📈 Training Configuration

Edit `params.yaml` to customize training:
```yaml
IMAGE_SIZE: [224, 224]
LEARNING_RATE: 0.0001
BATCH_SIZE: 16
EPOCHS: 50
AUGMENTATION: true
```

## 🧪 Testing
```bash
# Run unit tests
python -m pytest tests/

# Test prediction pipeline
python -c "from CNN_Classifier.components.prediction import PredictionPipeline; \
           predictor = PredictionPipeline('artifacts/training/model.h5'); \
           print(predictor.predict('path/to/image.jpg'))"
```

## 📊 Evaluation Metrics

After training, find evaluation results in `artifacts/evaluation/`:
- `evaluation_results.json` - Detailed metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curves.png` - ROC curves for all classes
- `class_distribution.png` - Class distribution comparison
- `sample_predictions.png` - Sample predictions with confidence

## 🛠️ Pipeline Stages

### Stage 1: Data Ingestion
Downloads and extracts CT scan dataset from Google Drive.

### Stage 2: Prepare Base Model
Loads ResNet50 with ImageNet weights and adds custom classification head.

### Stage 3: Model Training
Trains model with:
- Data augmentation
- Class weighting
- Early stopping
- Learning rate reduction
- Model checkpointing

### Stage 4: Model Evaluation
Generates comprehensive evaluation metrics and visualizations.

## 🎨 Model Architecture
```
Input (224x224x3)
    ↓
ResNet50 (Frozen)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization → Dense(1024) → Dropout(0.5)
    ↓
BatchNormalization → Dense(512) → Dropout(0.4)
    ↓
BatchNormalization → Dense(256) → Dropout(0.3)
    ↓
Dense(4, softmax)
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please contact: your.email@example.com

## 🙏 Acknowledgments

- Dataset: [Kidney CT Scan Dataset]
- Base Model: ResNet50 (ImageNet pre-trained)
- Framework: TensorFlow/Keras

## 📚 Citation

If you use this project in your research, please cite:
```bibtex
@software{kidney_disease_classifier,
  title={Kidney Disease Classification System},
  author={vivek kumar},
  year={2025},
  url={https://github.com/Vivekk-007/kidney-disease-classifier}
}
```

---

**⭐ Star this repo if you find it useful!**