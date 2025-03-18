import subprocess
import sys
import os
import glob
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
import shutil
import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA

# Install required packages
def install_packages():
    packages = ["torch", "torchvision", "torchsummary", "pandas", "scikit-learn", "rasterio", "matplotlib"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

install_packages()

# ------------------------
# Dataset Class
# ------------------------
class EurostatDataset(Dataset):
    def __init__(self, root_dir, labels_csv=None, transform=None, file_type="tif"):
        self.root_dir = root_dir
        self.transform = transform
        self.file_type = file_type  # "tif" for training, "npy" for testing
        self.images = []
        self.labels = []
        self.class_to_idx = {}  # Class label mapping

        if labels_csv:
            # For TEST SET (Uses CSV for labels)
            self.labels_df = pd.read_csv(labels_csv)
            self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.labels_df['label'].unique()))}

            for _, row in self.labels_df.iterrows():
                file_name = f"test_{row['test_id']}.npy"
                file_path = os.path.join(root_dir, file_name)
                if os.path.exists(file_path):
                    self.images.append(file_path)
                    self.labels.append(self.class_to_idx[row['label']])
        else:
            # For TRAIN SET (Auto-labels based on folder names)
            folders = sorted(os.listdir(root_dir))
            self.class_to_idx = {folder: idx for idx, folder in enumerate(folders)}

            for folder in folders:
                folder_path = os.path.join(root_dir, folder)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.endswith(f".{file_type}"):  # Match file type
                            self.images.append(os.path.join(folder_path, file))
                            self.labels.append(self.class_to_idx[folder])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import rasterio  # Ensure each worker imports rasterio
        img_path = self.images[idx]  
        label = self.labels[idx]

        if self.file_type == "tif":
            # Load .tif for training
            with rasterio.open(img_path) as src:
                image = src.read([1, 2, 3])  # Extract RGB bands
                image = np.moveaxis(image, 0, -1)  # Convert from (C, H, W) → (H, W, C)
        else:
            # Load .npy for testing
            image = np.load(img_path)[:, :, :3]  # Extract first 3 bands (RGB)

        # Normalize: Convert to float32 and scale to [0, 255]
        image = (image / image.max()) * 255.0  
        image = image.astype(np.uint8)

        # Convert to PIL image
        image = Image.fromarray(image)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

# ------------------------
# Main Script (Fix multiprocessing issue)
# ------------------------
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    BASE_DIR = os.getcwd() 

    # Load test dataset
    test_dataset = EurostatDataset(
        root_dir=os.path.join(BASE_DIR, "data", "testset"),
        labels_csv=os.path.join(BASE_DIR, "test_labels.csv"),
        transform=transform,
        file_type="npy"
    )

    print(f"Total samples in test_dataset: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)  # Set num_workers=0 to avoid Windows multiprocessing issue

    print(f"Total number of batches: {len(test_loader)}")
    print(f"Total samples: {len(test_loader.dataset)}")

    # ------------------------
    # Model Definition
    # ------------------------
    class EurosatModel(nn.Module):
        def __init__(self, num_classes=10):
            super(EurosatModel, self).__init__()
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        def forward(self, x):
            return self.model(x)

    model = EurosatModel(num_classes=10)

    # ------------------------
    # Training Setup
    # ------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ------------------------
    # Training Loop
    # ------------------------
    train_dataset = EurostatDataset(
        root_dir=os.path.join(BASE_DIR, "data", "EuroSATallBands", "ds", "images", "remote_sensing", "otherDatasets", "sentinel_2", "tif"),
        transform=transform,
        file_type="tif"
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)  # num_workers=0 to avoid multiprocessing issues

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        scheduler.step()

    print('Finished Training')

    # Save model
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_checkpoint_{current_time}.pth"
    torch.save(model.state_dict(), filename)

    # ------------------------
    # Evaluation Loop
    # ------------------------
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():  
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())

    print(f"First 10 test predictions: {predictions[:10]}")
    print(f'Test Accuracy: {100 * correct / total:.2f}%')
