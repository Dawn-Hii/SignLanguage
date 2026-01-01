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
from collections import deque, Counter

#CẤU HÌNH
MODEL_PATH = 'model_mobilenet.pth'
LABEL_PATH = 'label_map.pkl'
IMG_SIZE = 224

#CẤU HÌNH ĐỘ MƯỢT
CONFIDENCE_THRESHOLD = 0.85
SMOOTH_FACTOR = 0.8
PREDICTION_QUEUE_LEN = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#ĐỊNH NGHĨA MODEL
class MobileNetSignLanguage(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetSignLanguage, self).__init__()
        self.model = models.mobilenet_v2(weights=None)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)


# KHỞI TẠO
print("⏳ Đang tải tài nguyên...")

if not os.path.exists(LABEL_PATH):
    print(f"❌ Lỗi: Không tìm thấy file {LABEL_PATH}.")
    exit()
with open(LABEL_PATH, 'rb') as f:
    class_names = pickle.load(f)

model = MobileNetSignLanguage(len(class_names)).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Đã tải Model MobileNet thành công!")
else:
    print(f"❌ Lỗi: Không tìm thấy file {MODEL_PATH}")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.7)

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Queue lưu lịch sử dự đoán
predictions_queue = deque(maxlen=PREDICTION_QUEUE_LEN)

#CHẠY CAMERA
print("📷 Đang mở camera...")
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

            cx = int((min(x_list) + max(x_list)) / 2 * w)
            cy = int((min(y_list) + max(y_list)) / 2 * h)
            box_w = int((max(x_list) - min(x_list)) * w)
            box_h = int((max(y_list) - min(y_list)) * h)
            side_len = max(box_w, box_h) + 60

            target_xmin = max(0, cx - side_len // 2)
            target_ymin = max(0, cy - side_len // 2)
            target_xmax = min(w, cx + side_len // 2)
            target_ymax = min(h, cy + side_len // 2)

            if prev_coords is None:
                prev_coords = [float(target_xmin), float(target_ymin), float(target_xmax), float(target_ymax)]
            else:
                prev_coords[0] = prev_coords[0] * SMOOTH_FACTOR + target_xmin * (1 - SMOOTH_FACTOR)
                prev_coords[1] = prev_coords[1] * SMOOTH_FACTOR + target_ymin * (1 - SMOOTH_FACTOR)
                prev_coords[2] = prev_coords[2] * SMOOTH_FACTOR + target_xmax * (1 - SMOOTH_FACTOR)
                prev_coords[3] = prev_coords[3] * SMOOTH_FACTOR + target_ymax * (1 - SMOOTH_FACTOR)

            sx_min, sy_min = int(prev_coords[0]), int(prev_coords[1])
            sx_max, sy_max = int(prev_coords[2]), int(prev_coords[3])

            cv2.rectangle(frame, (sx_min, sy_min), (sx_max, sy_max), (0, 255, 0), 2)

            #DỰ ĐOÁN
            if sx_max > sx_min and sy_max > sy_min:
                crop = frame[sy_min:sy_max, sx_min:sx_max]
                if crop.size > 0:
                    try:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        enhanced_img = clahe.apply(gray)
                        blur = cv2.GaussianBlur(enhanced_img, (3, 3), 0)

                        pil_img = Image.fromarray(blur)
                        input_tensor = val_transform(pil_img).unsqueeze(0).to(device)

                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                            score, predicted = torch.max(probs, 1)

                            current_score = score.item()
                            predicted_char = class_names[predicted.item()]

                            # Cập nhật hàng đợi voting
                            if current_score > CONFIDENCE_THRESHOLD:
                                predictions_queue.append(predicted_char)
                            else:
                                predictions_queue.append("...")

                            # Voting
                            counter = Counter(predictions_queue)
                            most_common = counter.most_common(1)

                            if most_common:
                                top_label, count = most_common[0]
                                if count > (PREDICTION_QUEUE_LEN / 2):
                                    displayed_label = top_label
                                    # Lấy % của khung hình hiện tại để hiển thị cho sinh động
                                    current_pct = int(current_score * 100)

                            #HIỂN THỊ KẾT QUẢ ĐÃ SỬA
                            if displayed_label != "...":
                                color = (0, 255, 0)
                                # Đã thêm phần trăm vào đây
                                text = f"{displayed_label} ({current_pct}%)"
                            else:
                                color = (0, 0, 255)
                                text = "..."

                            cv2.rectangle(frame, (sx_min, sy_min - 40), (sx_max, sy_min), color, -1)
                            cv2.putText(frame, text, (sx_min + 10, sy_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    except:
                        pass
    else:
        if prev_coords is not None:
            pass
        prev_coords = None
        predictions_queue.clear()
        displayed_label = "..."

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Smoothed Sign Language", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
