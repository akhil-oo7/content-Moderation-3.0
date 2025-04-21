import cv2
import numpy as np
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, frame_interval=90, target_size=(112, 112)):
        """
        Initialize the VideoProcessor with optimized settings.
        
        Args:
            frame_interval (int): Number of frames to skip (increased to reduce processing)
            target_size (tuple): Target size for frame resizing (reduced for faster processing)
        """
        self.frame_interval = frame_interval
        self.target_size = target_size
    
    def extract_frames(self, video_path):
        """
        Extract frames from a video file efficiently.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            list: List of extracted frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Estimate video duration
        duration = total_frames / fps if fps > 0 else 0
        
        # Limit total frames to analyze based on duration
        max_frames_to_extract = min(30, int(total_frames / self.frame_interval))
        
        # Calculate frame positions to extract evenly throughout the video
        if total_frames > 0 and max_frames_to_extract > 0:
            frame_positions = [int(i * total_frames / max_frames_to_extract) for i in range(max_frames_to_extract)]
        else:
            frame_positions = []
        
        with tqdm(total=len(frame_positions), desc="Extracting frames") as pbar:
            for pos in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize frame to smaller target size
                    frame_resized = cv2.resize(frame_rgb, self.target_size)
                    frames.append(frame_resized)
                
                pbar.update(1)
        
        cap.release()
        return frames 