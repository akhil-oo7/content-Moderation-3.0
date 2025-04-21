# Content Moderation 3.0

A video content moderation system that uses machine learning to detect inappropriate content in videos.

## Features

- Upload videos through a simple web interface
- Extract frames from uploaded videos
- Analyze frames using a pre-trained neural network model
- Detect inappropriate content such as violence
- Display results with highlighted problematic frames

## Tech Stack

- Python
- Flask
- PyTorch
- OpenCV
- Transformers library

## Deployment

This application is configured for deployment on Render.

### Requirements

- Python 3.7+
- PyTorch
- A trained model in the `models/best_model` directory

## Usage

1. Upload a video through the web interface
2. Wait for the analysis to complete
3. Review the results showing safe/unsafe content

## Model

The application uses a fine-tuned ResNet-50 model trained to detect inappropriate content in video frames. 