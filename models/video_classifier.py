import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
import tqdm
import json
import random
import io
import base64
from torch.utils.data.sampler import WeightedRandomSampler

class VideoDataset(Dataset):
    """Dataset for accident/non-accident videos with augmentation"""
    def __init__(self, root_dir, transform=None, sequence_length=16, limit=None, augment=False, validation=False):
        self.transform = transform
        self.sequence_length = sequence_length
        self.augment = augment and not validation
        self.validation = validation
        self.videos = []
        self.labels = []
        self.paths = []
        
        # Load accident videos (label 1)
        accident_dir = os.path.join(root_dir, 'Accident')
        if os.path.exists(accident_dir):
            files = os.listdir(accident_dir)
            if limit:
                files = files[:limit]
            for filename in files:
                if filename.endswith(('.mp4', '.avi', '.mov')):
                    self.videos.append(os.path.join(accident_dir, filename))
                    self.labels.append(1)
                    self.paths.append(os.path.join(accident_dir, filename))
        
        # Load non-accident videos (label 0)
        non_accident_dir = os.path.join(root_dir, 'Non Accident')
        if os.path.exists(non_accident_dir):
            files = os.listdir(non_accident_dir)
            if limit:
                files = files[:limit]
            for filename in files:
                if filename.endswith(('.mp4', '.avi', '.mov')):
                    self.videos.append(os.path.join(non_accident_dir, filename))
                    self.labels.append(0)
                    self.paths.append(os.path.join(non_accident_dir, filename))
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        path = self.paths[idx]
        
        # Extract frames from video
        try:
            frames, random_frame = self._extract_frames(video_path)
            
            # Apply augmentation if enabled (only for training)
            if self.augment and not self.validation and random.random() < 0.7:
                frames = self._augment_frames(frames)
            
            if self.transform:
                frames = [self.transform(frame) for frame in frames]
                random_frame_tensor = self.transform(random_frame)
            
            frames_tensor = torch.stack(frames)
            return frames_tensor, label, path, random_frame_tensor
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Return blank frames if there's an error
            if self.transform:
                blank = Image.new('RGB', (224, 224), (0, 0, 0))
                blank_frames = [self.transform(blank) for _ in range(self.sequence_length)]
                blank_tensor = torch.stack(blank_frames)
                return blank_tensor, label, path, blank_frames[0]
            return torch.zeros((self.sequence_length, 3, 224, 224)), label, path, torch.zeros((3, 224, 224))
    
    def _augment_frames(self, frames):
        """Apply consistent augmentation across all frames"""
        augmented_frames = []
        
        # Random horizontal flip
        if random.random() < 0.5:
            frames = [frame.transpose(Image.FLIP_LEFT_RIGHT) for frame in frames]
        
        # Random brightness and contrast adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        
        for frame in frames:
            # Apply brightness and contrast adjustments
            enhancer = transforms.ColorJitter(brightness=brightness_factor, contrast=contrast_factor)
            augmented_frame = enhancer(frame)
            augmented_frames.append(augmented_frame)
        
        return augmented_frames
    
    def _extract_frames(self, video_path):
        """Extract a sequence of frames from a video and a random frame for display"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # If video has fewer frames than sequence_length, duplicate frames
        if frame_count <= self.sequence_length:
            indices = list(range(frame_count)) * (self.sequence_length // frame_count + 1)
            indices = indices[:self.sequence_length]
        else:
            # Sample frames evenly
            indices = np.linspace(0, frame_count - 1, self.sequence_length, dtype=int)
        
        # Select a random frame index for display
        random_frame_idx = random.randint(0, frame_count - 1) if frame_count > 0 else 0
        
        frames = []
        random_frame = None
        
        # First extract the random frame for display
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
        ret, frame = cap.read()
        if ret:
            random_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            random_frame = Image.fromarray(random_frame)
        else:
            random_frame = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Then extract the sequence frames
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                frame = Image.fromarray(frame)
                frames.append(frame)
            else:
                # If frame read failed, add a blank frame
                frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))
        
        cap.release()
        return frames, random_frame

class TemporalCNN(nn.Module):
    """Enhanced CNN with temporal features for video classification"""
    def __init__(self, num_classes=2):
        super(TemporalCNN, self).__init__()
        
        # Load pre-trained ResNet-50 as base model for better feature extraction
        self.base_model = resnet50(weights='DEFAULT')
        
        # Remove the final fully connected layer
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Freeze early layers but allow fine-tuning of later layers for better transfer learning
        ct = 0
        for child in self.base_model.children():
            ct += 1
            if ct < 7:  # Freeze early layers
                for param in child.parameters():
                    param.requires_grad = False
        
        # Add temporal processing layers with improved architecture
        self.lstm = nn.LSTM(
            input_size=2048,  # ResNet-50 outputs 2048 features
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5  # Increased dropout for better regularization
        )
        
        # Enhanced attention mechanism with multi-head attention
        self.attention_heads = 4
        self.attention_dim = 256
        
        # Create multiple attention heads
        self.attention_queries = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, self.attention_dim),  # 1024 from bidirectional LSTM (512*2)
                nn.Tanh(),
                nn.Linear(self.attention_dim, 1),
            ) for _ in range(self.attention_heads)
        ])
        
        # Final classification layer with enhanced architecture
        self.fc = nn.Sequential(
            nn.Linear(1024 * self.attention_heads, 1024),  # Concatenate all attention heads
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape for CNN processing
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features using CNN
        x = self.base_model(x)
        
        # Reshape for LSTM processing
        x = x.view(batch_size, seq_len, -1)
        
        # Process temporal features
        x, _ = self.lstm(x)  # x shape: [batch_size, seq_len, hidden_size*2]
        
        # Apply multi-head attention mechanism
        multi_head_context = []
        for i in range(self.attention_heads):
            # Calculate attention weights for this head
            attention_logits = self.attention_queries[i](x)  # shape: [batch_size, seq_len, 1]
            attention_weights = torch.softmax(attention_logits, dim=1)  # shape: [batch_size, seq_len, 1]
            
            # Apply attention weights to get context vector for this head
            context = torch.sum(attention_weights * x, dim=1)  # shape: [batch_size, hidden_size*2]
            multi_head_context.append(context)
        
        # Concatenate all attention heads
        context_vector = torch.cat(multi_head_context, dim=1)  # shape: [batch_size, hidden_size*2 * num_heads]
        
        # Final classification
        output = self.fc(context_vector)
        
        return output

class VideoClassifier:
    def __init__(self, model_path=None):
        self.img_height = 224  # ResNet expects 224x224 images
        self.img_width = 224
        self.sequence_length = 16  # Number of frames to sample from each video
        self.model_path = model_path if model_path else 'models/saved/video_model.pth'
        self.best_model_path = 'models/saved/video_model_best.pth'
        
        # Create directories if they don't exist
        os.makedirs('models/saved', exist_ok=True)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Define transforms for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create model
        self.model = TemporalCNN(num_classes=2).to(self.device)
        
        # Load model if it exists
        if os.path.exists(self.best_model_path):
            print(f"Loading best model from {self.best_model_path}")
            try:
                self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            except Exception as e:
                print(f"Error loading best model: {e}")
                print("Training a new model with ResNet50 architecture...")
                self.train_model()
        elif os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Training a new model with ResNet50 architecture...")
                self.train_model()
        else:
            print("No pre-trained model found. Training a new model...")
            self.train_model()
    
    def extract_frames(self, video_path):
        """Extract frames from a video file for prediction with random frame for display"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return None, None
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None, None
        
        # Calculate frames to skip to get sequence_length frames
        if total_frames <= self.sequence_length:
            # If video has fewer frames than needed, duplicate frames
            indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        else:
            # Otherwise, sample frames evenly
            indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        
        # Select a random frame index for display
        random_frame_idx = random.randint(0, total_frames - 1) if total_frames > 0 else 0
        
        # First extract the random frame for display
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
        ret, frame = cap.read()
        if ret:
            random_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            random_frame_pil = Image.fromarray(random_frame)
            random_frame_tensor = self.transform(random_frame_pil)
        else:
            random_frame_tensor = torch.zeros((3, self.img_height, self.img_width))
        
        # Extract sequence frames for prediction
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_img = Image.fromarray(frame)
                # Apply transform
                tensor_img = self.transform(pil_img)
                frames.append(tensor_img)
            else:
                # If frame reading fails, add a blank frame
                blank = torch.zeros((3, self.img_height, self.img_width))
                frames.append(blank)
        
        cap.release()
        return torch.stack(frames), random_frame_tensor
    
    def train_model(self, epochs=10, batch_size=4):
        """Train the model on accident/non-accident videos with validation"""
        # Create datasets with data augmentation
        full_dataset = VideoDataset(
            root_dir='/home/debian/Documents/Accident/videos',
            transform=self.transform,
            sequence_length=self.sequence_length,
            limit=None,  # Use all available videos
            augment=True  # Enable augmentation
        )
        
        if len(full_dataset) == 0:
            print("No training data found!")
            return
        
        # Split into training and validation sets
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        indices = list(range(len(full_dataset)))
        train_indices, val_indices = indices[:train_size], indices[train_size:]
        
        # Create subset samplers
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        
        # Count class distribution for logging
        class_counts = [0, 0]
        for _, label, _, _ in full_dataset:
            class_counts[label] += 1
        
        # Create data loaders with appropriate samplers
        train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0)
        
        print(f"Training with {train_size} videos, validating with {val_size} videos")
        print(f"Class distribution: Accident: {class_counts[1]}, Non-Accident: {class_counts[0]}")
        
        # Calculate class weights for balanced loss
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float).to(self.device)
        if class_counts[0] > 0 and class_counts[1] > 0:
            weight_ratio = class_counts[0] / class_counts[1]
            if weight_ratio > 1:
                class_weights = torch.tensor([1.0, weight_ratio], dtype=torch.float).to(self.device)
            else:
                class_weights = torch.tensor([1.0/weight_ratio, 1.0], dtype=torch.float).to(self.device)
        
        # Define loss function with class weights and label smoothing for better generalization
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        
        # Two-phase training approach
        print("Phase 1: Training classifier only...")
        
        # First phase: Train only the classifier layers
        for name, param in self.model.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Optimizer with appropriate learning rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                               lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                                steps_per_epoch=len(train_loader), 
                                                epochs=5, pct_start=0.3)
        
        # Track best validation accuracy
        best_val_acc = 0.0
        
        # Phase 1: Train only the classifier for a few epochs
        phase1_epochs = 5
        for epoch in range(phase1_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{phase1_epochs} [Train Phase 1]")
            for inputs, labels, _, _ in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'accuracy': 100 * correct / total
                })
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels, _, _ in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{phase1_epochs} [Val Phase 1]"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            print(f"Epoch {epoch+1}/{phase1_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        # Phase 2: Fine-tune the model with LSTM and attention layers
        print("\nPhase 2: Fine-tuning LSTM and attention layers...")
        
        # Load the best model from phase 1
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        
        # Unfreeze LSTM and attention layers
        for name, param in self.model.named_parameters():
            if 'fc' in name or 'lstm' in name or 'attention' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # New optimizer with lower learning rate for fine-tuning
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                              lr=5e-5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-phase1_epochs)
        
        # Reset best validation accuracy for phase 2
        best_val_acc_phase2 = best_val_acc
        patience = 3
        no_improvement = 0
        
        # Phase 2 training loop
        for epoch in range(phase1_epochs, epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train Phase 2]")
            for inputs, labels, _, _ in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'accuracy': 100 * correct / total
                })
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            scheduler.step()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels, _, _ in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val Phase 2]"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping and model saving
            if val_acc > best_val_acc_phase2:
                best_val_acc_phase2 = val_acc
                best_val_acc = max(best_val_acc, best_val_acc_phase2)
                no_improvement = 0
                
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print("Early stopping triggered.")
                    break
        
        # Save the final model
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Final model saved to {self.model_path}")
        print(f"Best model saved to {self.best_model_path} with validation accuracy: {best_val_acc:.2f}%")
        
        # Load the best model
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
    
    def predict_with_frame(self, video_path):
        """Predict whether a video contains an accident or not and return a random frame"""
        try:
            # Extract frames from the video and get a random frame for display
            frames, random_frame = self.extract_frames(video_path)
            if frames is None:
                return "Error", 0.0, None
            
            # Add batch dimension
            frames = frames.unsqueeze(0).to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(frames)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item() * 100  # Convert to percentage
            
            # Convert random frame tensor to base64 for web display
            random_frame = random_frame.permute(1, 2, 0).cpu().numpy()
            # Denormalize
            random_frame = random_frame * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            random_frame = np.clip(random_frame, 0, 1) * 255
            random_frame = random_frame.astype(np.uint8)
            
            # Convert to PIL Image
            random_frame_pil = Image.fromarray(random_frame)
            
            # Convert to base64
            buffered = io.BytesIO()
            random_frame_pil.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Return result, confidence, and frame
            result = "Accident" if predicted_class == 1 else "Non Accident"
            return result, confidence, img_str
        except Exception as e:
            print(f"Error predicting video {video_path}: {e}")
            return "Error", 0.0, None
    
    def predict(self, video_path):
        """Predict whether a video contains an accident or not"""
        try:
            # Extract frames from the video
            frames, _ = self.extract_frames(video_path)
            if frames is None:
                return "Error", 0.0
            
            # Add batch dimension
            frames = frames.unsqueeze(0).to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(frames)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item() * 100  # Convert to percentage
            
            # Return result and confidence
            if predicted_class == 1:
                return "Accident", confidence
            else:
                return "Non Accident", confidence
        except Exception as e:
            print(f"Error predicting video {video_path}: {e}")
            return "Error", 0.0
