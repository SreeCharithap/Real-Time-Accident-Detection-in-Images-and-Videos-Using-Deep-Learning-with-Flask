# 🚗 Accident Detection System

This is a deep learning-based **Accident Detection System** built using **Flask**, **PyTorch**, and advanced CNN-LSTM architectures. It can classify both images and videos as either containing an "Accident" or "Non Accident".

## 🎥 Demo Video

<video width="640" height="360" controls>
  <source src="Accident Detection System.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

> You can also [download the demo video](Accident%20Detection%20System.mp4) to view it locally.

## 🧠 Models Used

### 1. `image_classifier.py`
A **ResNet-18** based model trained to detect accidents from images.

#### Features:
- Uses pre-trained ResNet-18 architecture
- Custom classifier head
- Supports Grad-CAM visualization
- High accuracy on test data

### 2. `video_classifier.py`
A hybrid **CNN + LSTM + Attention** model based on ResNet-50 for video classification.

#### Features:
- Temporal modeling using BiLSTM
- Multi-head attention mechanism
- Frame sampling and augmentation
- Returns confidence scores and sample frames

## 🖥️ App Structure

Built using **Flask**, this web application provides a user-friendly interface for uploading and classifying:
- Static images (`jpg`, `png`)
- Videos (`mp4`, `avi`, `mov`)

It returns predictions along with confidence scores and visualizations where applicable.

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- PyTorch installed with CUDA support (if available)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/accident-detection-system.git    
   cd accident-detection-system