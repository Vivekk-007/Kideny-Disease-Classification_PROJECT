from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from CNN_Classifier.components.prediction import PredictionPipeline
from CNN_Classifier import logger

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'artifacts/training/model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize prediction pipeline
try:
    predictor = PredictionPipeline(MODEL_PATH)
    logger.info("Prediction pipeline initialized successfully")
except Exception as e:
    logger.error(f"Error initializing prediction pipeline: {e}")
    predictor = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if predictor is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = predictor.predict(filepath)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify(result)
        
        return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, or PNG'}), 400
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    }), 200


@app.route('/info')
def info():
    """Model information endpoint"""
    return jsonify({
        'model': 'Kidney Disease Classifier',
        'version': '1.0.0',
        'classes': ['Cyst', 'Normal', 'Stone', 'Tumor'],
        'input_size': '224x224',
        'description': 'Deep learning model for kidney disease classification from CT scans'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)