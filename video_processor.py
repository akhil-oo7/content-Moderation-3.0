import cv2

class VideoProcessor:
    def __init__(self, frame_interval=30, target_size=(224, 224)):
        """
        Initialize the VideoProcessor.
        frame_interval: Extract one frame every N frames
        target_size: Resize frames to this size
        """
        self.frame_interval = frame_interval
        self.target_size = target_size
    
    def extract_frames(self, video_path):
        """
        Extract frames from a video file.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame to target size
                frame_resized = cv2.resize(frame_rgb, self.target_size)
                frames.append(frame_resized)
            
            frame_count += 1
        
        cap.release()
        return frames 