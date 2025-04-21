from flask import Flask, render_template, request, jsonify, url_for, Response, stream_with_context
import os
from werkzeug.utils import secure_filename
from video_processor import VideoProcessor
from content_moderator import ContentModerator
import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import json
from torch.utils.data import DataLoader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['FRAMES_FOLDER'] = 'static/frames'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FRAMES_FOLDER'], exist_ok=True)

# Initialize components with trained model
video_processor = VideoProcessor()
content_moderator = ContentModerator(train_mode=False)  # Use trained model

# Configure logging
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

def frame_to_base64(frame):
    """Convert a numpy array frame to base64 encoded string"""
    # Convert numpy array to PIL Image
    image = Image.fromarray(frame)
    # Create a bytes buffer
    buffer = BytesIO()
    # Save image to buffer in JPEG format
    image.save(buffer, format='JPEG')
    # Encode buffer to base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{img_str}'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process video and get frames
            frames = video_processor.extract_frames(filepath)
            
            # Create a folder to save frame images
            frames_folder = os.path.join(app.config['FRAMES_FOLDER'], os.path.splitext(filename)[0])
            os.makedirs(frames_folder, exist_ok=True)
            
            # Save frames as images
            frame_paths = []
            for i, frame in enumerate(frames):
                frame_path = os.path.join(frames_folder, f"frame_{i}.jpg")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_paths.append(os.path.join('static/frames', os.path.splitext(filename)[0], f"frame_{i}.jpg"))
            
            # Analyze frames for content moderation
            results = content_moderator.analyze_frames(frames)
            
            # Calculate overall video safety
            unsafe_frames = [r for r in results if r['flagged']]
            total_frames = len(results)
            unsafe_percentage = (len(unsafe_frames) / total_frames) * 100
            
            # Prepare response
            response = {
                'status': 'UNSAFE' if unsafe_frames else 'SAFE',
                'total_frames': total_frames,
                'unsafe_frames': len(unsafe_frames),
                'unsafe_percentage': unsafe_percentage,
                'confidence': 1.0 if not unsafe_frames else max(r['confidence'] for r in unsafe_frames),
                'details': []
            }
            
            # Add details for unsafe frames with frame images
            if unsafe_frames:
                for i, result in enumerate(results):
                    if result['flagged']:
                        response['details'].append({
                            'frame_index': i,
                            'confidence': result['confidence'],
                            'reason': result['reason'],
                            'frame_image_url': frame_paths[i]  # Add the image URL
                        })
            
            # Clean up original video file
            if os.path.exists(filepath):
                os.remove(filepath)
                
            return jsonify(response)
            
        except Exception as e:
            app.logger.error(f"Error processing video: {str(e)}")
            return jsonify({'error': str(e)}), 500

# Add error handling
@app.errorhandler(Exception)
def handle_error(error):
    app.logger.error(f'An error occurred: {str(error)}')
    return jsonify({'error': 'An internal server error occurred'}), 500

if __name__ == '__main__':
    # Only run in debug mode locally
    app.run(debug=True)
else:
    # In production
    app.run(host='0.0.0.0') 