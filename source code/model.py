import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import numpy as np
import cv2
import face_recognition
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ResNet50 + LSTM model for deepfake detection
class ResNet50LSTM(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(ResNet50LSTM, self).__init__()
        resnet = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional, batch_first=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if len(x.shape) != 5:
            raise ValueError(f"Expected input to have 5 dimensions, but got {x.shape}")

        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.resnet(x)
        x = self.avgpool(x)
        x = x.view(batch_size * seq_length, -1)
        x = x.view(batch_size, seq_length, -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the last LSTM output
        return x

# Dataset class for video frames
class VideoDataset(Dataset):
    def __init__(self, video_paths, sequence_length=20, transform=None):
        self.video_paths = video_paths
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = []
        for frame in self.extract_frames(video_path):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except IndexError:
                continue  # Skip if no face found
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            if len(frames) == self.sequence_length:
                break

        # Ensure we have exactly sequence_length frames
        if len(frames) < self.sequence_length:
            frames += [torch.zeros((3, 112, 112))] * (self.sequence_length - len(frames))

        frames = torch.stack(frames)  # Stack frames into a tensor
        label = 1 if "real" in video_path else 0  # Assuming the file paths determine the class
        return frames, label

    def extract_frames(self, path):
        vid_obj = cv2.VideoCapture(path)
        total_frames = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the indices of the frames to be extracted
        frame_indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)

        for idx in frame_indices:
            vid_obj.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the frame position
            success, image = vid_obj.read()
            if success and image is not None and isinstance(image, np.ndarray) and image.size != 0:
                yield image
            else:
                break  # Stop if there are no more frames to read

# Function to load model checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=device)
    model = ResNet50LSTM(num_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Evaluation function to calculate predictions
def evaluate_model(model,dataloader):
    model.eval()  # Set the model to evaluation mode
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running_loss = 0.0
    all_probabilities = []  # List to hold the probabilities

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Calculate loss (ensure criterion is defined)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            all_probabilities.append(probabilities.cpu().numpy())  # Store probabilities

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    
    # Convert the list of probabilities to a numpy array
    all_probabilities = np.concatenate(all_probabilities)

    print(f'Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}')
    return all_probabilities  # Return the probabilities
