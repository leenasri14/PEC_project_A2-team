from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import torch
import torchvision
import numpy as np
import cv2
import face_recognition
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F
import random
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix, classification_report,roc_auc_score, log_loss
import seaborn as sns
import matplotlib.pyplot as plt # Add precision_score


# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded files
app.secret_key = 'supersecretkey'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Create folder if not exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Make sure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Model definition (same as the previous code)
class ResNet50LSTM(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(ResNet50LSTM, self).__init__()
        resnet = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional, batch_first=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.resnet(x)
        x = self.avgpool(x)
        x = x.view(batch_size * seq_length, -1)
        x = x.view(batch_size, seq_length, -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the last LSTM output
        return x

# Dataset class for testing video frames
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
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                print("Invalid frame, skipping...")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
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

        if len(frames) < self.sequence_length:
            frames += [torch.zeros((3, 112, 112))] * (self.sequence_length - len(frames))

        frames = torch.stack(frames)  # Stack all the frames
        label = 0  # No labels are needed for the test
        return frames, label

    def extract_frames(self, path):
        vid_obj = cv2.VideoCapture(path)
        total_frames = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)

        for idx in frame_indices:
            vid_obj.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, image = vid_obj.read()
            if success :
                yield image
            else:
                break
    

# Load model from checkpoint
#def load_checkpoint(filepath, model):
#    checkpoint = torch.load(filepath, map_location='cuda:0')
#    model.load_state_dict(checkpoint['model_state_dict'])
#   return model
def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))  # Force CPU if CUDA unavailable
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
# Evaluate model and return result
def evaluate_video(video_path):
    model.eval()  # Set the model to evaluation mode
    test_dataset = VideoDataset([video_path], sequence_length=30, transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for inputs, _ in test_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)  # Get probabilities for each class

            # Determine the predicted label based on the probabilities
            predicted_label = 'fake' if probabilities[0][0] > 0.5 else 'real'
            
            # Return both the predicted label and probabilities
            return predicted_label, probabilities.cpu().numpy() 

# Allowed file check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Transforms
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load model and checkpoint
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ResNet50LSTM(num_classes=2).to(device)
checkpoint_path = 'C:/Users/leena/OneDrive/Documents/III-CSE-A BATCH-1/resnet50_lstm_epoch3.pth'
model = load_checkpoint(checkpoint_path, model)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    # Handle file upload
    return "File uploaded!"

@app.route('/random_testing', methods=['GET', 'POST'])
def random_testing():
    if request.method == 'POST':
        real_folder_path = request.form['real_folder_path']
        fake_folder_path = request.form['fake_folder_path']
        num_videos = int(request.form['num_videos'])

        # Check if the provided folders exist
        if os.path.exists(real_folder_path) and os.path.exists(fake_folder_path):
            # Get the video files from both folders
            real_videos = [f for f in os.listdir(real_folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
            fake_videos = [f for f in os.listdir(fake_folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

            # Combine and shuffle videos, selecting random ones
            total_videos = real_videos[:] + fake_videos[:]
            selected_videos = random.sample(total_videos, min(num_videos, len(total_videos)))

            results = []
            actual_classes = []
            predicted_classes = []
            probabilities_list = []

            for video in selected_videos:
                if video in real_videos:
                    video_path = os.path.join(real_folder_path, video)
                    actual_class = "real"
                else:
                    video_path = os.path.join(fake_folder_path, video)
                    actual_class = "fake"

                # Perform prediction on the video
                predicted_class, probabilities = evaluate_video(video_path)

                # Skip video if no predictions were made
                if predicted_class is None:
                    continue

                # Append the results
                results.append({
                    'video': video,
                    'actual': actual_class,
                    'predicted': predicted_class
                })
                actual_classes.append(actual_class)
                predicted_classes.append(predicted_class)
                probabilities_list.append(probabilities)

            # Calculate performance metrics
            metrics = calculate_metrics(actual_classes, predicted_classes)

            # Convert actual and predicted classes to numeric for AUC and Log Loss
            actual_numeric = [1 if label == 'real' else 0 for label in actual_classes]
            predicted_numeric = [1 if label == 'real' else 0 for label in predicted_classes]

            # Reshape probabilities array and calculate AUC and Log Loss
            probabilities_array = np.array(probabilities_list).reshape(-1, 2)
            auc_and_loss_metrics = calculate_auc_and_log_loss(probabilities_array, actual_numeric)

            # Save confusion matrix plot
            confusion_matrix_path = 'static/confusion_matrix.png'
            plot_confusion_matrix(metrics['confusion_matrix'], confusion_matrix_path)

            return render_template('random_testing_results.html', 
                                   results=results, 
                                   metrics=metrics, 
                                   auc_and_loss=auc_and_loss_metrics, 
                                   report=metrics['classification_report'])

        else:
            flash('The specified real or fake folder does not exist.')
            return redirect(request.url)

    return render_template('random_testing.html')

@app.route('/testing', methods=['GET', 'POST'])
def testing():
    predictions = {}

    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        files = request.files.getlist('files[]')

        if not files:
            flash('No selected file')
            return redirect(request.url)
        
        # Loop through the uploaded files
        for file in files:
            if file.filename == '':
                flash('No selected file')
                continue

            if file:
                # Save the file to the upload folder
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                # Perform deepfake detection using evaluate_video on the uploaded video
                predicted_class, probabilities = evaluate_video(filepath)
                predictions[file.filename] = {
                    'prediction': predicted_class,
                    'probabilities': probabilities
                }
        
        # Render the template with predictions
        return render_template('testing.html', predictions=predictions)
    
    return render_template('testing.html')


def plot_confusion_matrix(conf_matrix, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Real', 'Predicted Fake'],
                yticklabels=['Actual Real', 'Actual Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()


def calculate_auc_and_log_loss(probabilities_array, actual_numeric):
    # Ensure probabilities_array has the correct shape and actual_numeric contains both classes
    if len(set(actual_numeric)) < 2:  # Check if both classes are present
        return {'auc': 'N/A', 'log_loss': 'N/A'}  # Return 'N/A' or None if AUC cannot be computed

    # AUC Calculation
    auc = roc_auc_score(actual_numeric, probabilities_array[:, 1])  # Assuming 'fake' class is the second column

    # Log Loss Calculation
    log_loss_value = log_loss(actual_numeric, probabilities_array)

    return {'auc': auc, 'log_loss': log_loss_value}


def calculate_metrics(actual, predicted):
    # Precision, Recall, F1 Score for both classes
    precision_macro = precision_score(actual, predicted, average='macro')
    recall_macro = recall_score(actual, predicted, average='macro')
    f1_macro = f1_score(actual, predicted, average='macro')

    precision_weighted = precision_score(actual, predicted, average='weighted')
    recall_weighted = recall_score(actual, predicted, average='weighted')
    f1_weighted = f1_score(actual, predicted, average='weighted')
    
    # Precision, Recall, F1 Score for each class
    precision_per_class = precision_score(actual, predicted, average=None, labels=['real', 'fake'])
    recall_per_class = recall_score(actual, predicted, average=None, labels=['real', 'fake'])
    f1_per_class = f1_score(actual, predicted, average=None, labels=['real', 'fake'])

    # Confusion Matrix
    conf_matrix = confusion_matrix(actual, predicted, labels=['real', 'fake'])

    # Classification report
    #class_report = classification_report(actual, predicted, labels=['real', 'fake'], target_names=['real', 'fake'])

   # Accuracy
    accuracy = accuracy_score(actual, predicted)

    class_report = classification_report(actual, predicted, output_dict=True, labels=['real', 'fake'])

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report  # Ensure this is included
    }
# Call the function with your actual and predicted data



if __name__ == "__main__":
    app.run(debug=True)