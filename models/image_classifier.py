import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import timm
from tqdm import tqdm
import cv2
import shutil
import base64
import io
import json
import random

# Paths to dataset
accident_dir = r"/home/debian/Documents/Accident/Images/Accident"
non_accident_dir = r"/home/debian/Documents/Accident/Images/Non Accident"

# Supported image extensions
supported_extensions = ['.png', '.jpg', '.jpeg']

# Function to load images
def load_images(accident_dir, non_accident_dir):
    image_paths = []
    labels = []
    for label, directory in enumerate([non_accident_dir, accident_dir]):  # 0 for non-accident, 1 for accident
        if os.path.exists(directory):
            for img in os.listdir(directory):
                ext = os.path.splitext(img.lower())[1]
                if ext in supported_extensions:
                    image_paths.append(os.path.join(directory, img))
                    labels.append(label)
    return image_paths, labels

class AccidentDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            print(f"Error loading image: {img_path} - {e}")
            return torch.zeros((3, 224, 224)), label, img_path  # Return dummy tensor on failure

class GradCAM:
    """Class for generating Grad-CAM visualization"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, x, class_idx=None):
        # Forward pass
        logits = self.model(x)
        
        # Get the target class
        if class_idx is None:
            _, predicted = torch.max(logits, 1)
            target_class = predicted.item()
        else:
            target_class = class_idx
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Compute weights
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU and normalization
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.cpu().numpy(), target_class

# Simple and effective model using ResNet18
from torchvision.models import resnet18

class AccidentDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, device="cuda"):
        super(AccidentDetector, self).__init__()
        self.device = device

        # Load pre-trained ResNet18 for feature extraction
        self.base_model = resnet18(pretrained=pretrained)
        
        # Freeze early layers but allow fine-tuning of later layers
        for name, param in self.base_model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        
        # Replace classifier head with a simple but effective one
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

class ImageClassifier:
    def __init__(self, model_path=None):
        """Initialize the image classifier"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Define model paths
        self.model_path = model_path or os.path.join("models", "saved", "image_model.pth")
        self.best_model_path = os.path.join("models", "saved", "image_model_best.pth")
        
        # Create directories if they don't exist
        os.makedirs('models/saved', exist_ok=True)
        
        # Create model
        self.model = AccidentDetector(num_classes=2, pretrained=True, device=self.device).to(self.device)
        
        # Try to load pre-trained model
        try:
            if os.path.exists(self.best_model_path):
                print(f"Loading best model from {self.best_model_path}")
                self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
                print("Model loaded successfully")
            else:
                print("No pre-trained model found. Training a new model...")
                self._train_and_save_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model...")
            self._train_and_save_model()
        
        # Initialize GradCAM for visualization
        try:
            # For ResNet18, the last layer is layer4
            self.grad_cam = GradCAM(self.model, self.model.base_model.layer4[-1])
        except Exception as e:
            print(f"Could not initialize GradCAM: {e}")
            self.grad_cam = None
    
    def _train_and_save_model(self):
        """Train and save the model using a simple and effective approach"""
        # Load image paths and labels
        image_paths, labels = load_images(accident_dir, non_accident_dir)
        
        if len(image_paths) == 0:
            print("No training data found!")
            return
        
        # Split into train, validation, test sets
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        print(f"Training set: {len(train_paths)} images")
        print(f"Validation set: {len(val_paths)} images")
        print(f"Test set: {len(test_paths)} images")
        
        # Define transformations with effective but not excessive augmentation
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = AccidentDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = AccidentDataset(val_paths, val_labels, transform=val_transform)
        test_dataset = AccidentDataset(test_paths, test_labels, transform=val_transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        # Define loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer with appropriate learning rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Simple learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                           factor=0.5, patience=3)
        
        # Train the model
        best_val_accuracy = 0.0
        patience = 7
        no_improvement = 0
        num_epochs = 30
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
            
            for images, labels, _ in train_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                train_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
            
            train_accuracy = 100 * correct / total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            
            with torch.no_grad():
                for images, labels, _ in val_bar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    val_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
            
            val_accuracy = 100 * correct / total
            
            # Update learning rate scheduler based on validation accuracy
            scheduler.step(val_accuracy)
            
            print(f"\nEpoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            # Save model if it's the best so far
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                no_improvement = 0
                
                # Save the model
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Best model saved at {self.best_model_path}")
                print(f"✅ File saved successfully. Size: {os.path.getsize(self.best_model_path)} bytes")
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print("Early stopping triggered.")
                    break
        
        # Evaluate on test set with the best model
        print("\nLoading best model for evaluation...")
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_accuracy = 100 * correct / total
        print(f"\nTest Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Calculate additional metrics
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        
        # Save final model (same as best model)
        shutil.copy(self.best_model_path, self.model_path)
        print(f"Best model copied to {self.model_path}")
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    
    def predict_with_heatmap(self, image_path):
        """Predict whether an image contains an accident or not and generate heatmap"""
        try:
            # Define transform for prediction
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item() * 100
            
            # Generate class activation map
            if self.grad_cam is not None:
                # Generate heatmap
                input_tensor.requires_grad = True
                cam, _ = self.grad_cam(input_tensor, class_idx=predicted_class)
                cam = cam.squeeze()
                
                # Resize CAM to match original image size
                cam_resized = cv2.resize(cam, (image.width, image.height))
                
                # Convert to heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Overlay heatmap on original image
                img_array = np.array(image)
                superimposed = heatmap * 0.4 + img_array * 0.6
                superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
                
                # Convert to base64 for display
                _, buffer = cv2.imencode('.png', superimposed)
                heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
            else:
                heatmap_base64 = None
            
            # Convert original image to base64 for display
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Return prediction and heatmap
            result = {
                'class': 'Accident' if predicted_class == 1 else 'Non Accident',
                'confidence': confidence,
                'heatmap': heatmap_base64,
                'original_image': img_base64
            }
            return result
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {'error': str(e)}
    
    def predict(self, image_path):
        """Predict whether an image contains an accident or not"""
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item() * 100  # Convert to percentage
            
            # Return result and confidence
            if predicted_class == 1:
                return "Accident", confidence
            else:
                return "Non Accident", confidence
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return "Error", 0.0
