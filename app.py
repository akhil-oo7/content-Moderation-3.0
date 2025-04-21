from flask import Flask, render_template, request, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from video_processor import VideoProcessor
from content_moderator import ContentModerator
import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import time
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Initialize components with trained model (only once at startup)
print("Initializing components...")
video_processor = VideoProcessor(frame_interval=90, target_size=(112, 112))
content_moderator = ContentModerator(train_mode=False)
print("Components initialized")

def frame_to_base64(frame):
    """Convert a numpy array frame to base64 encoded string"""
    try:
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame)
        # Create a bytes buffer
        buffer = BytesIO()
        # Save image to buffer in JPEG format with compression
        image.save(buffer, format='JPEG', quality=80)
        # Encode buffer to base64
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f'data:image/jpeg;base64,{img_str}'
    except Exception as e:
        app.logger.error(f"Error converting frame to base64: {str(e)}")
        return None

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
        
        try:
            # Save uploaded file
            file.save(filepath)
            app.logger.info(f"Processing video: {filename}")
            
            start_time = time.time()
            
            # Process video and get frames
            frames = video_processor.extract_frames(filepath)
            app.logger.info(f"Extracted {len(frames)} frames in {time.time() - start_time:.2f} seconds")
            
            if not frames:
                return jsonify({'error': 'No frames could be extracted from the video'}), 400
            
            # Analyze frames for content moderation
            analysis_start = time.time()
            results = content_moderator.analyze_frames(frames)
            app.logger.info(f"Analysis completed in {time.time() - analysis_start:.2f} seconds")
            
            # Calculate overall video safety
            unsafe_frames = [r for r in results if r['flagged']]
            total_frames = len(results)
            unsafe_percentage = (len(unsafe_frames) / total_frames) * 100 if total_frames > 0 else 0
            
            # Prepare response
            response = {
                'status': 'UNSAFE' if unsafe_frames else 'SAFE',
                'total_frames': total_frames,
                'unsafe_frames': len(unsafe_frames),
                'unsafe_percentage': unsafe_percentage,
                'confidence': 1.0 if not unsafe_frames else max(r['confidence'] for r in unsafe_frames),
                'details': []
            }
            
            # Add details and base64 images for unsafe frames
            if unsafe_frames:
                encoding_start = time.time()
                for i, result in enumerate(results):
                    if result['flagged']:
                        # Convert frame to base64
                        img_data = frame_to_base64(frames[i])
                        
                        if img_data:
                            response['details'].append({
                                'frame': i,
                                'reason': result['reason'],
                                'confidence': result['confidence'],
                                'image_data': img_data
                            })
                app.logger.info(f"Encoded {len(unsafe_frames)} frames in {time.time() - encoding_start:.2f} seconds")
            
            # Clean up uploaded file
            os.remove(filepath)
            
            total_time = time.time() - start_time
            app.logger.info(f"Total processing time: {total_time:.2f} seconds")
            
            return jsonify(response)
            
        except Exception as e:
            app.logger.error(f"Error processing video: {str(e)}")
            # Clean up uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    app.logger.error(f"404 error: {request.url}")
    return jsonify(error=str(e)), 404

@app.errorhandler(Exception)
def handle_error(error):
    app.logger.error(f"An error occurred: {str(error)}")
    return jsonify({'error': 'An internal server error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
else:
    # In production
    app.run(host='0.0.0.0') 