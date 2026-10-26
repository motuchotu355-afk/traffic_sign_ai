import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO

# --- 1. Sign Names Map ---
SIGN_NAMES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles > 3.5t', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: 'Vehicles > 3.5t prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
    28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow',
    31: 'Wild animals crossing', 32: 'End of all speed & passing limits',
    33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only',
    36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles > 3.5t'
}

# --- 2. PyTorch Classifier Model ---
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

# --- 3. Preprocessing Functions ---
def apply_clahe(np_rgb_crop):
    img_lab = cv2.cvtColor(np_rgb_crop, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB))

# --- 4. Main Two-Stage Pipeline ---
def run_pipeline(image_path, model_path="traffic_sign_model.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Step A: Load Classifier
    classifier = TrafficSignNet(num_classes=43).to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval()

    # Step B: Load YOLOv8 Detector
    # Using yolov8n.pt (COCO pretrained: class 9 is 'traffic light', class 11 is 'stop sign')
    # Or replace with a custom traffic-sign detector weights file if available
    detector = YOLO("yolov8n.pt")

    # Load original image
    bgr_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb_img.shape

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    # Run detection
    results = detector(rgb_img, conf=0.25)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)

    annotated = bgr_img.copy()

    if len(boxes) == 0:
        print("No traffic signs or objects detected by YOLO. Trying fallback center crop...")
        # Fallback: center crop if detector misses
        min_dim = min(h, w)
        cx, cy = w // 2, h // 2
        boxes = [[cx - min_dim//4, cy - min_dim//4, cx + min_dim//4, cy + min_dim//4]]

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Ensure coordinates stay within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            continue

        # Extract only the sign region
        crop_rgb = rgb_img[y1:y2, x1:x2]
        crop_clahe = apply_clahe(crop_rgb)

        tensor_crop = transform(crop_clahe).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = classifier(tensor_crop)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class]) * 100
        sign_label = SIGN_NAMES.get(pred_class, f"Class {pred_class}")

        # Draw bounding box and label on output
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        caption = f"{sign_label} ({confidence:.1f}%)"
        cv2.putText(annotated, caption, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"Detection {i+1}: {sign_label} | Confidence: {confidence:.2f}% | Box: [{x1}, {y1}, {x2}, {y2}]")

    output_path = "detected_output.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"\nResult saved to: {output_path}")

if __name__ == "__main__":
    import sys
    img_target = sys.argv[1] if len(sys.argv) > 1 else "test_road.jpg"
    if os.path.exists(img_target):
        run_pipeline(img_target)
    else:
        print(f"Image '{img_target}' not found. Please provide a valid file path.")