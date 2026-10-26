import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# --- GTSRB Class Name Dictionary ---
SIGN_NAMES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# --- Network Architecture (Must match train.py) ---
class TrafficSignNet(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# --- Helper Functions ---
@st.cache_resource
def load_model(weights_path="traffic_sign_model.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TrafficSignNet(num_classes=43).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        return model, device
    return None, device

def apply_clahe(pil_img):
    img = np.array(pil_img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_rgb)

# --- Streamlit Layout ---
st.set_page_config(page_title="Traffic Sign Recognizer", page_icon="🚦", layout="centered")

st.title("🚦 Traffic Sign Recognition System")
st.write("Upload a traffic sign image or pick a sample from the GTSRB dataset.")

model, device = load_model()

if model is None:
    st.error("Error: `traffic_sign_model.pth` not found in current folder. Run train.py first!")
    st.stop()

# --- Sidebar: Sample Dataset Image Picker ---
st.sidebar.header("Dataset Samples")
use_sample = st.sidebar.checkbox("Load a test image from GTSRB dataset")
selected_img_path = None

if use_sample:
    test_csv_path = "./gtsrb/Test.csv"
    if os.path.exists(test_csv_path):
        df_test = pd.read_csv(test_csv_path)
        sample_options = [
            f"Index {idx}: Class {row['ClassId']} ({SIGN_NAMES.get(row['ClassId'], 'Unknown')})"
            for idx, row in df_test.head(30).iterrows()
        ]
        choice = st.sidebar.selectbox("Choose sample image:", sample_options)
        selected_index = int(choice.split(":")[0].replace("Index ", ""))
        selected_img_path = os.path.join("./gtsrb", df_test.iloc[selected_index]['Path'])
    else:
        st.sidebar.warning("`./gtsrb/Test.csv` not found.")

# --- Confidence Threshold Slider ---
confidence_threshold = st.sidebar.slider(
    "Uncertainty / Confidence Threshold (%)",
    min_value=30.0,
    max_value=99.0,
    value=75.0,
    step=5.0,
    help="Flags predictions with maximum confidence below this threshold as Out-of-Distribution or ambiguous."
)

# --- Image Input Selection ---
uploaded_file = st.file_uploader(
    "Upload custom image (.png, .jpg, .jpeg, .ppm)...", 
    type=["png", "jpg", "jpeg", "ppm"],
    disabled=use_sample
)

target_image = None
if use_sample and selected_img_path and os.path.exists(selected_img_path):
    target_image = Image.open(selected_img_path).convert("RGB")
elif uploaded_file is not None:
    target_image = Image.open(uploaded_file).convert("RGB")

# --- Inference and Visual Display ---
if target_image is not None:
    clahe_image = apply_clahe(target_image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(target_image, use_container_width=True)
    with col2:
        st.subheader("CLAHE Equalized")
        st.image(clahe_image, use_container_width=True)

    # Standardize to 32x32 and apply GTSRB normalization
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    tensor_input = transform(clahe_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_input)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    top3_indices = probs.argsort()[-3:][::-1]
    top_class = int(top3_indices[0])
    top_prob = float(probs[top_class]) * 100

    st.markdown("---")

    # Handle In-Distribution vs Out-of-Distribution / Unknown signs
    if top_prob < confidence_threshold:
        st.warning(
            f"⚠️ **Low Confidence / Unrecognized Sign:** Model is only `{top_prob:.1f}%` confident it is "
            f"**{SIGN_NAMES.get(top_class, f'Class {top_class}')}**.\n\n"
            f"This sign may be outside the 43 GTSRB benchmark categories (e.g., U-Turn, custom street names) "
            f"or visually degraded."
        )
    else:
        st.success(
            f"✅ **Recognized Sign:** {SIGN_NAMES.get(top_class, f'Class {top_class}')} (`{top_prob:.2f}%` Confidence)"
        )

    # Breakdown of top 3 classes
    st.subheader("Top 3 Candidates")
    for idx in top3_indices:
        class_name = SIGN_NAMES.get(idx, f"Class {idx}")
        percentage = float(probs[idx]) * 100
        st.write(f"**{class_name}**: `{percentage:.2f}%`")
        st.progress(percentage / 100.0)