import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
import time
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

class VideoFrameDataset(Dataset):
    def __init__(self, frames, labels, feature_extractor):
        self.frames = frames
        self.labels = labels
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame)
        
        # Preprocess the image
        inputs = self.feature_extractor(image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ContentModerator:
    def __init__(self, model_name="microsoft/resnet-50", train_mode=False):
        """
        Initialize the ContentModerator with a pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            train_mode (bool): Whether to initialize in training mode
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Load model only once and keep in memory
        print(f"Initializing model on {self.device}...")
        start_time = time.time()
        
        # Always use feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        if train_mode:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=2,  # Binary classification: violent vs non-violent
                ignore_mismatched_sizes=True
            ).to(self.device)
        else:
            # Load our trained model
            model_path = os.path.join("models", "best_model")
            if os.path.exists(model_path):
                print("Loading trained model...")
                self.model = AutoModelForImageClassification.from_pretrained(
                    model_path,
                    num_labels=2
                ).to(self.device)
                self.model.eval()  # Set to evaluation mode
            else:
                raise FileNotFoundError("Trained model not found. Please train the model first.")
        
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    def analyze_frames(self, frames):
        """
        Analyze frames for inappropriate content with optimized batch processing.
        
        Args:
            frames (list): List of video frames as numpy arrays
            
        Returns:
            list: List of analysis results for each frame
        """
        if not frames:
            return []
            
        results = []
        
        # Convert frames to dataset
        dataset = VideoFrameDataset(frames, [0] * len(frames), self.feature_extractor)
        
        # Use larger batch size for faster processing
        batch_size = min(64, len(frames))
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        start_time = time.time()
        print(f"Analyzing {len(frames)} frames with batch size {batch_size}...")
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Analyzing frames"):
                pixel_values = batch['pixel_values'].to(self.device)
                outputs = self.model(pixel_values)
                predictions = torch.softmax(outputs.logits, dim=1)
                
                for pred in predictions:
                    # Get probability of violence (class 1)
                    violence_prob = pred[1].item()
                    # Lower threshold for violence detection
                    flagged = violence_prob > 0.3  # Changed from 0.5 to 0.3
                    
                    results.append({
                        'flagged': flagged,
                        'reason': "Detected violence" if flagged else "No inappropriate content detected",
                        'confidence': violence_prob if flagged else 1 - violence_prob
                    })
        
        print(f"Analysis completed in {time.time() - start_time:.2f} seconds")
        return results 