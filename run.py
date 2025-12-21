import cv2
import numpy as np
import mediapipe as mp

# Tự động chọn thư viện TFLite có sẵn trên máy
import tensorflow.lite as tflite

# --- CẤU HÌNH QUAN TRỌNG ---
MODEL_PATH = 'sign_language.tflite' # Tên file model .tflite bạn đã tải về
IMG_SIZE = 224

# ✅ DANH SÁCH LỚP CỦA BẠN (Đã cập nhật đúng thứ tự)
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']

# --- LOAD MODEL TFLITE ---
print("Dang tai model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Lấy thông tin cổng vào/ra của model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model da san sang! Hay dua tay vao camera.")

# Khởi tạo MediaPipe để bắt bàn tay
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,              # Chỉ bắt 1 tay
    min_detection_confidence=0.7, # Độ tin cậy > 70% mới bắt
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # 1. Lật ảnh như gương
    frame = cv2.flip(frame, 1)
    
    # 2. MediaPipe cần ảnh RGB để tìm tay
    img_rgb_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb_mp)
    
    h, w, c = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # --- Tìm tọa độ khung bao quanh bàn tay ---
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x); x_max = max(x_max, x)
                y_min = min(y_min, y); y_max = max(y_max, y)
            
            # Nới rộng khung ra một chút (Padding 40px)
            padding = 40
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            
            # Vẽ khung xanh lá cây
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            try:
                # --- CẮT & XỬ LÝ ẢNH ĐỂ ĐƯA VÀO MODEL ---
                img_crop = frame[y_min:y_max, x_min:x_max]
                
                if img_crop.size != 0:
                    # A. Chuyển BGR (OpenCV) -> RGB (Model MobileNet yêu cầu)
                    # Nếu thiếu dòng này, model sẽ bị loạn màu và đoán sai!
                    img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                    
                    # B. Resize về đúng kích thước 224x224
                    img_resize = cv2.resize(img_crop, (IMG_SIZE, IMG_SIZE))
                    
                    # C. Chuẩn hóa dữ liệu về khoảng [-1, 1]
                    # Công thức: (Pixel / 127.5) - 1.0
                    img_array = img_resize.astype(np.float32)
                    input_data = (img_array / 127.5) - 1.0 
                    
                    # D. Thêm chiều batch (1, 224, 224, 3)
                    input_data = np.expand_dims(input_data, axis=0)
                    
                    # --- DỰ ĐOÁN ---
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    
                    # Lấy kết quả cao nhất
                    index = np.argmax(output_data)
                    confidence = output_data[0][index]
                    
                    # Chỉ hiện chữ nếu độ tin cậy > 80%
                    if confidence > 0.8:
                        label = f"{CLASS_NAMES[index]} ({confidence*100:.0f}%)"
                        # Hiện chữ màu vàng trên đầu khung
                        cv2.putText(frame, label, (x_min, y_min - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        # Nếu không chắc chắn thì hiện ???
                        cv2.putText(frame, "???", (x_min, y_min - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                   
            except Exception as e:
                pass # Bỏ qua lỗi cắt ảnh nếu tay ra khỏi khung hình

    # Hiện màn hình
    cv2.imshow("Nhan dien ngon ngu ky hieu", frame)
    
    # Bấm phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()