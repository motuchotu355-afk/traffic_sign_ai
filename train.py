import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- 1. Dataset Loader with CLAHE Equalization ---
class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def apply_clahe(self, img_path):
        img = cv2.imread(img_path)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Instantiate CLAHE locally to avoid multiprocessing serialization errors
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_rgb)

    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx]['Path']
        full_path = os.path.join(self.root_dir, rel_path)
        
        image = self.apply_clahe(full_path)
        label = int(self.df.iloc[idx]['ClassId'])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# --- 2. Traffic Sign CNN Architecture ---
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

# --- 3. Training Execution ---
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using compute device: {device}")

    train_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    train_set = GTSRBDataset('./gtsrb/Train.csv', './gtsrb', transform=train_transforms)
    
    # num_workers=0 runs data loading in the primary process to prevent pickling issues
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

    model = TrafficSignNet(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 10
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_acc = (correct / total) * 100
        avg_loss = total_loss / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), "traffic_sign_model.pth")
    print("\nSaved weights to traffic_sign_model.pth")

if __name__ == '__main__':
    main()