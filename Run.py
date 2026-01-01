import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
import pickle
import os
import time  # <--- Import thư viện thời gian
from PIL import Image

# --- 1. CẤU HÌNH ---
MODEL_PATH = 'model_mobilenet.pth'
LABEL_PATH = 'label_map.pkl'
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.85
SMOOTH_FACTOR = 0.7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 2. ĐỊNH NGHĨA MODEL ---
class MobileNetSignLanguage(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetSignLanguage, self).__init__()
        self.model = models.mobilenet_v2(weights=None)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)


# --- 3. KHỞI TẠO ---
print("⏳ Đang tải tài nguyên...")

# Load nhãn
if not os.path.exists(LABEL_PATH):
    print(f"❌ Lỗi: Không tìm thấy file {LABEL_PATH}.")
    exit()
with open(LABEL_PATH, 'rb') as f:
    class_names = pickle.load(f)

# Load Model
model = MobileNetSignLanguage(len(class_names)).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Đã tải Model MobileNet thành công!")
else:
    print(f"❌ Lỗi: Không tìm thấy file {MODEL_PATH}")
    exit()

# MediaPipe & Transform
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.7)

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# CLAHE (Cân bằng sáng)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# --- 4. CHẠY CAMERA ---
print("📷 Đang mở camera...")
cap = cv2.VideoCapture(0)

prev_coords = None
current_label = "..."
prev_time = 0  # <--- Biến lưu thời gian khung hình trước

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Lấy tọa độ
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            # Tính khung vuông
            cx = int((min(x_list) + max(x_list)) / 2 * w)
            cy = int((min(y_list) + max(y_list)) / 2 * h)
            box_w = int((max(x_list) - min(x_list)) * w)
            box_h = int((max(y_list) - min(y_list)) * h)
            side_len = max(box_w, box_h) + 60

            x_min = max(0, cx - side_len // 2)
            y_min = max(0, cy - side_len // 2)
            x_max = min(w, cx + side_len // 2)
            y_max = min(h, cy + side_len // 2)

            # Làm mượt
            if prev_coords is None:
                prev_coords = [x_min, y_min, x_max, y_max]
            else:
                prev_coords[0] = int(prev_coords[0] * SMOOTH_FACTOR + x_min * (1 - SMOOTH_FACTOR))
                prev_coords[1] = int(prev_coords[1] * SMOOTH_FACTOR + y_min * (1 - SMOOTH_FACTOR))
                prev_coords[2] = int(prev_coords[2] * SMOOTH_FACTOR + x_max * (1 - SMOOTH_FACTOR))
                prev_coords[3] = int(prev_coords[3] * SMOOTH_FACTOR + y_max * (1 - SMOOTH_FACTOR))

            sx_min, sy_min, sx_max, sy_max = prev_coords
            cv2.rectangle(frame, (sx_min, sy_min), (sx_max, sy_max), (0, 255, 0), 2)

            # --- DỰ ĐOÁN ---
            try:
                crop = frame[sy_min:sy_max, sx_min:sx_max]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    enhanced_img = clahe.apply(gray)
                    blur = cv2.GaussianBlur(enhanced_img, (3, 3), 0)

                    cv2.imshow("AI Input (CLAHE)", cv2.resize(blur, (200, 200)))

                    pil_img = Image.fromarray(blur)
                    input_tensor = val_transform(pil_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        score, predicted = torch.max(probs, 1)

                        current_score = score.item()

                        if current_score > CONFIDENCE_THRESHOLD:
                            current_label = f"{class_names[predicted.item()]} ({int(current_score * 100)}%)"
                            color = (0, 255, 0)
                        else:
                            current_label = "..."
                            color = (0, 0, 255)

                        cv2.rectangle(frame, (sx_min, sy_min - 40), (sx_max, sy_min), color, -1)
                        cv2.putText(frame, current_label, (sx_min + 5, sy_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except:
                pass
    else:
        prev_coords = None

    # --- TÍNH TOÁN & HIỆN FPS ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Vẽ FPS lên góc trái trên
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("MobileNet Hand Sign (FPS)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()