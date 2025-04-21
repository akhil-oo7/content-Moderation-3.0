import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
from safetensors.torch import load_file  # Assuming you have a library to load safetensors

class VideoFrameDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame)
        
        # Preprocess the image (custom preprocessing if needed)
        image = image.resize((224, 224))  # Example resize, adjust as needed
        image = np.array(image).astype(np.float32) / 255.0  # Normalize
        image = torch.tensor(image).permute(2, 0, 1)  # Convert to tensor and change dimension order
        
        return {
            'pixel_values': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

class ContentModerator:
    def __init__(self, train_mode=False):
        """
        Initialize the ContentModerator with a pre-trained model.
        
        Args:
            train_mode (bool): Whether to initialize in training mode
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load our trained model
        model_path = os.path.join("models", "best_model", "model.safetensors")
        if os.path.exists(model_path):
            print("Loading trained model...")
            state_dict = load_file(model_path)  # Load the state dictionary
            self.model = YourModelClass()  # Replace with your model class
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        else:
            raise FileNotFoundError("Trained model not found. Please train the model first.")
    
    def analyze_frames(self, frames):
        """
        Analyze frames for inappropriate content.
        
        Args:
            frames (list): List of video frames as numpy arrays
            
        Returns:
            list: List of analysis results for each frame
        """
        results = []
        
        # Convert frames to dataset
        dataset = VideoFrameDataset(frames, [0] * len(frames))
        dataloader = DataLoader(dataset, batch_size=32)
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch['pixel_values'].to(self.device)
                outputs = self.model(pixel_values)
                predictions = torch.softmax(outputs, dim=1)  # Assuming model outputs logits
                
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
        
        return results 