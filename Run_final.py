import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
import pickle
import os
import time
from PIL import Image
from collections import deque, Counter, OrderedDict

# CẤU HÌNH
MODEL_PATH = 'model_mobilenet_stable.pth' 
LABEL_PATH = 'label_map.pkl'       
IMG_SIZE = 224

# CẤU HÌNH ĐỘ MƯỢT
CONFIDENCE_THRESHOLD = 0.8
SMOOTH_FACTOR = 0.7         
PREDICTION_QUEUE_LEN = 8    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#MODEL
class MobileNetSignLanguage(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetSignLanguage, self).__init__()
        # Load khung
        self.model = models.mobilenet_v2(weights=None)
        
        # Sửa Input thành 1 kênh (Grayscale)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Sửa Classifier y hệt lúc train
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        return self.model(x)

#KHỞI TẠO
print("Đang tải tài nguyên...")

#Load Nhãn
if not os.path.exists(LABEL_PATH):
    print(f"Lỗi: Thiếu file {LABEL_PATH}")
    exit()
with open(LABEL_PATH, 'rb') as f:
    class_names = pickle.load(f)

#Load Model & Fix lỗi module.
model = MobileNetSignLanguage(len(class_names)).to(device)

if os.path.exists(MODEL_PATH):
    # Load state dict
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # Xử lý key 'module.'
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") 
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Đã tải Model thành công!")
else:
    print(f"Lỗi: Thiếu file {MODEL_PATH}")
    exit()

#MediaPipe & Transform
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.5)

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), #1 kênh
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
predictions_queue = deque(maxlen=PREDICTION_QUEUE_LEN)

# CHẠY CAMERA
print("📷 Đang mở camera... (Nhấn 'q' để thoát)")
cap = cv2.VideoCapture(0)

prev_coords = None
displayed_label = "..."
current_pct = 0 
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            cx, cy = int(np.mean(x_list) * w), int(np.mean(y_list) * h)
            box_w = int((max(x_list) - min(x_list)) * w)
            box_h = int((max(y_list) - min(y_list)) * h)
            max_side = max(box_w, box_h)


            side_len = int(max_side * 1.3)

            target_xmin = max(0, cx - side_len // 2)
            target_ymin = max(0, cy - side_len // 2)
            target_xmax = min(w, cx + side_len // 2)
            target_ymax = min(h, cy + side_len // 2)

            if prev_coords is None:
                prev_coords = [float(target_xmin), float(target_ymin), float(target_xmax), float(target_ymax)]
            else:
                for i, val in enumerate([target_xmin, target_ymin, target_xmax, target_ymax]):
                    prev_coords[i] = prev_coords[i] * SMOOTH_FACTOR + val * (1 - SMOOTH_FACTOR)

            sx_min, sy_min, sx_max, sy_max = [int(v) for v in prev_coords]
            cv2.rectangle(frame, (sx_min, sy_min), (sx_max, sy_max), (0, 255, 0), 2)

            if sx_max > sx_min and sy_max > sy_min:
                crop = frame[sy_min:sy_max, sx_min:sx_max]
                if crop.size > 0:
                    try:
                        # Tiền xử lý
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        enhanced = clahe.apply(gray)
                        blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
                        
                        # Dự đoán
                        pil_img = Image.fromarray(blur)
                        input_tensor = val_transform(pil_img).unsqueeze(0).to(device)

                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                            score, predicted = torch.max(probs, 1)

                            if score.item() > CONFIDENCE_THRESHOLD:
                                predictions_queue.append(class_names[predicted.item()])
                            else:
                                predictions_queue.append("...")

                            # Voting
                            most_common = Counter(predictions_queue).most_common(1)
                            if most_common:
                                top_label, count = most_common[0]
                                if count > (PREDICTION_QUEUE_LEN / 2):
                                    displayed_label = top_label
                                    current_pct = int(score.item() * 100)

                            # Hiển thị
                            text = "..." if displayed_label == "..." else f"{displayed_label} ({current_pct}%)"
                            color = (0, 0, 255) if displayed_label == "..." else (0, 255, 0)
                            
                            cv2.rectangle(frame, (sx_min, sy_min - 40), (sx_max, sy_min), color, -1)
                            cv2.putText(frame, text, (sx_min + 10, sy_min - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    except: pass
    else:
        prev_coords = None
        predictions_queue.clear()
        displayed_label = "..."

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Sign Language AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()