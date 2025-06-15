from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import io
import base64
from werkzeug.utils import secure_filename
import numpy as np
from models.image_classifier import ImageClassifier
from models.video_classifier import VideoClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'videos'), exist_ok=True)

# Initialize classifiers
image_classifier = ImageClassifier()
video_classifier = VideoClassifier()

def allowed_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_IMAGE_EXTENSIONS']

def allowed_video(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_VIDEO_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/classify_image', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_image(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'images', filename)
        file.save(filepath)
        
        # Classify the image with heatmap visualization
        result_dict = image_classifier.predict_with_heatmap(filepath)
        
        if 'error' in result_dict:
            return jsonify({'error': result_dict['error']})
        
        # The result_dict now contains base64 encoded images
        return jsonify({
            'result': result_dict['class'],
            'confidence': float(result_dict['confidence']),
            'filename': filename,
            'original_image': result_dict['original_image'],
            'heatmap_image': result_dict['heatmap']
        })
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/classify_video', methods=['POST'])
def classify_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_video(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'videos', filename)
        file.save(filepath)
        
        # Classify the video and get a random frame
        result, confidence, random_frame_b64 = video_classifier.predict_with_frame(filepath)
        
        return jsonify({
            'result': result,
            'confidence': float(confidence),
            'filename': filename,
            'random_frame': random_frame_b64
        })
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)
