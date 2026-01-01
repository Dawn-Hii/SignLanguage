import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import pickle
import matplotlib.pyplot as plt
import copy
import numpy as np

# --- 1. CẤU HÌNH (ĐÃ TINH CHỈNH) ---
DATA_DIR = 'DataSet_ThanhBinh'
MODEL_PATH = 'model_mobilenet_stable.pth'
LABEL_PATH = 'label_map.pkl'
IMG_SIZE = 224

# ✅ 1. Tăng Batch Size để đường Loss mượt hơn (Trung bình hóa tốt hơn)
BATCH_SIZE = 64
# ✅ 2. Giảm Learning Rate để model học chậm mà chắc
LEARNING_RATE = 0.0003

EPOCHS = 100

PATIENCE = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model():
    print(f"🚀 Đang chạy trên thiết bị: {device}")

    if not os.path.exists(DATA_DIR):
        print("❌ LỖI: Không tìm thấy thư mục dữ liệu!")
        return

    # --- 2. DATA AUGMENTATION MẠNH HƠN ---
    # ✅ 3. Thêm biến đổi ảnh để model không học vẹt
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),  # Xoay nghiêng +/- 15 độ
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Dịch chuyển ảnh
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Đổi độ sáng/tương phản
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Validation không cần Data Augmentation (chỉ cần resize và chuẩn hóa)
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    full_dataset = datasets.ImageFolder(root=DATA_DIR)  # Load thô trước
    class_names = full_dataset.classes
    print(f"✅ Class: {class_names}")

    # Chia tập dữ liệu
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Gán transform riêng cho từng tập
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform
    # (Lưu ý: cách gán trên hơi hack, chuẩn nhất là tạo class Dataset riêng,
    # nhưng với ImageFolder cơ bản thì ta dùng transform trong DataLoader như sau là ổn nhất:)

    # Cách chuẩn hơn cho ImageFolder khi chia split:
    # Ta tạo lại 2 dataset riêng biệt trỏ cùng folder nhưng khác transform
    train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)

    # Dùng indices của random_split để lấy đúng mẫu
    train_dataset = torch.utils.data.Subset(train_dataset, train_subset.indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_subset.indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    with open(LABEL_PATH, 'wb') as f:
        pickle.dump(class_names, f)

    # --- 3. MODEL VỚI DROPOUT ---
    class MobileNetSignLanguage(nn.Module):
        def __init__(self, num_classes):
            super(MobileNetSignLanguage, self).__init__()
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

            # ✅ 5. Thêm Dropout vào Classifier để chống Overfitting
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),  # Ngẫu nhiên tắt 30% nơ-ron
                nn.Linear(1280, num_classes)
            )

        def forward(self, x):
            return self.model(x)

    model = MobileNetSignLanguage(len(class_names)).to(device)

    # --- 4. OPTIMIZER & LOSS ---
    # ✅ 4. Label Smoothing giúp Loss mượt hơn
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Weight Decay giúp giữ trọng số nhỏ, tránh biến động mạnh
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    min_val_loss = np.inf
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("🔥 Bắt đầu Train ổn định...")

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = 100 * val_correct / val_total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        scheduler.step(epoch_val_loss)
        curr_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch + 1}/{EPOCHS}] LR:{curr_lr:.6f} | "
              f"Train Loss:{epoch_loss:.4f} Acc:{epoch_acc:.1f}% | "
              f"Val Loss:{epoch_val_loss:.4f} Acc:{epoch_val_acc:.1f}%")

        if epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("   --> 💾 Saved Best Model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("🛑 Early Stopping.")
                break

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss (Smoothed)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy (Smoothed)')
    plt.legend()

    plt.savefig('training_stable.png')
    plt.show()


if __name__ == '__main__':
    train_model()