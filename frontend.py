import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import pickle as pk

# --- 1. Define the Model Architecture ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 2. Load Model Artifacts ---
import torch.serialization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")

PKL_PATH = os.path.join(ARTIFACTS_DIR, "model_artifacts.pkl")
WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "best_model_state_dict.pth")

if not os.path.exists(PKL_PATH) or not os.path.exists(WEIGHTS_PATH):
    st.error("Model artifacts not found. Please check directory structure.")
    st.stop()

# 1. Override PyTorch's internal restore location to force CPU
orig_restore_location = torch.serialization.default_restore_location

def cpu_restore_location(storage, location):
    return storage.cpu()

try:
    torch.serialization.default_restore_location = cpu_restore_location
    with open(PKL_PATH, "rb") as f:
        artifact = pk.load(f)
finally:
    # Reset it back to normal after loading
    torch.serialization.default_restore_location = orig_restore_location

# Extract metadata
num_classes = artifact['num_classes']
idx_to_class = artifact['idx_to_class']
IMAGE_SIZE = artifact['image_size']
MEAN = artifact['normalize_mean']
STD = artifact['normalize_std']

# 2. Setup Device & Instantiate Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=num_classes).to(device)

# 3. Load State Dict
state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# --- 3. Image Transformations ---
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# --- 4. Prediction Function ---
def predict_image(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        conf, predicted = torch.max(probabilities, 1)

    return idx_to_class[predicted.item()], conf.item()

# --- 5. Streamlit UI ---
st.title("Intel Image Classification")
st.write("Upload an image or use your webcam to classify environmental scenes (buildings, forest, glacier, mountain, sea, street).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Take a picture")

image_to_predict = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    if opencv_image is not None:
        opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        image_to_predict = Image.fromarray(opencv_image_rgb)
        st.image(image_to_predict, caption="Uploaded Image", use_container_width=True)
    else:
        st.error("Failed to decode uploaded image. Please try another file.")

elif camera_input is not None:
    camera_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(camera_bytes, 1)
    
    if opencv_image is not None:
        opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        image_to_predict = Image.fromarray(opencv_image_rgb)
        st.image(image_to_predict, caption="Captured Image", use_container_width=True)
    else:
        st.error("Failed to decode camera image. Please try again.")

if image_to_predict is not None:
    with st.spinner('Classifying...'):
        class_name, confidence = predict_image(image_to_predict)
        st.success(f"Prediction: **{class_name}** with confidence **{confidence:.2f}**")