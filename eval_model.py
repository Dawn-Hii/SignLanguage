import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import os
import numpy as np

#CẤU HÌNH
DATA_DIR = 'DataSet_ThanhBinh'  # Thư mục dữ liệu
MODEL_PATH = 'model_pytorch.pth'
LABEL_PATH = 'label_map.pkl'
IMG_SIZE = 224
BATCH_SIZE = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#MODEL
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


#CHẠY ĐÁNH GIÁ
def evaluate():
    print("Đang tải dữ liệu kiểm tra...")

    # Load Label
    with open(LABEL_PATH, 'rb') as f:
        class_names = pickle.load(f)

    # Load Model
    model = SignLanguageCNN(len(class_names)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Chuẩn bị dữ liệu
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    print("Đang chấm điểm model...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Vẽ Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('AI Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
    plt.show()

    # Tính độ chính xác tổng
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"Độ chính xác tổng quát: {acc * 100:.2f}%")


if __name__ == "__main__":
    evaluate()