from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from video_processor import VideoProcessor
from content_moderator import ContentModerator
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
video_processor = VideoProcessor()
content_moderator = ContentModerator()

def frame_to_base64(frame):
    """Convert a numpy array frame to base64 encoded string"""
    image = Image.fromarray(frame)
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
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
            
            # Analyze frames for content moderation
            results = content_moderator.analyze_frames(frames)
            
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
            
            # Add details for unsafe frames
            if unsafe_frames:
                for i, result in enumerate(results):
                    if result['flagged']:
                        # Convert frame to base64
                        img_data = frame_to_base64(frames[i])
                        
                        response['details'].append({
                            'frame': i,
                            'reason': result['reason'],
                            'confidence': result['confidence'],
                            'image_data': img_data
                        })
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(response)
            
        except Exception as e:
            # Clean up uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

# Use environment variable for port (Render sets this)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port) 