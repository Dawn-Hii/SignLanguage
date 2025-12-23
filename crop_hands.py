import cv2
import os
import mediapipe as mp
import math


INPUT_DIR = r"D:\Train\AI\SignLanguage\dataset_sign_language"  
OUTPUT_DIR = r"D:\Train\AI\SignLanguage\dataset_cropped"       
TARGET_SIZE = 224   
PADDING = 50        

# Khởi tạo bộ nhận diện tay của MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

def crop_hand_from_image(image_path, save_path):
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Nếu tìm thấy tay
    if results.multi_hand_landmarks:
        h, w, c = img.shape
        
        # Lấy tọa độ các khớp tay của bàn tay đầu tiên
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Tìm khung bao quanh (Bounding Box)
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y
            
        # --- XỬ LÝ CẮT HÌNH VUÔNG (QUAN TRỌNG) ---
        # Để ảnh không bị méo khi resize, ta phải cắt hình vuông ngay từ đầu
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        # Tìm cạnh lớn nhất để làm cạnh hình vuông
        max_side = max(box_w, box_h) + PADDING * 2
        
        # Tính tâm của bàn tay
        center_x = x_min + box_w // 2
        center_y = y_min + box_h // 2
        
        # Tính lại toạ độ cắt mới (hình vuông mở rộng từ tâm)
        new_x_min = max(0, center_x - max_side // 2)
        new_y_min = max(0, center_y - max_side // 2)
        new_x_max = min(w, center_x + max_side // 2)
        new_y_max = min(h, center_y + max_side // 2)
        
        # Cắt ảnh
        crop_img = img[new_y_min:new_y_max, new_x_min:new_x_max]
        
        
        try:
            final_img = cv2.resize(crop_img, (TARGET_SIZE, TARGET_SIZE))
            cv2.imwrite(save_path, final_img)
            return True
        except Exception as e:
            print(f"Lỗi khi resize ảnh {image_path}: {e}")
            return False
            
    return False


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Đang bắt đầu xử lý cắt ảnh...")
processed_count = 0
skipped_count = 0

# Duyệt qua từng folder con (A, B, C...)
for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            # Tạo đường dẫn ảnh gốc
            full_input_path = os.path.join(root, file)
            
            # Tạo đường dẫn ảnh đích (giữ nguyên cấu trúc thư mục)
            relative_path = os.path.relpath(root, INPUT_DIR) # Lấy phần đuôi (ví dụ: \A)
            target_folder = os.path.join(OUTPUT_DIR, relative_path)
            
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            
            full_output_path = os.path.join(target_folder, file)
            
            # Thực hiện cắt
            success = crop_hand_from_image(full_input_path, full_output_path)
            
            if success:
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"Đã xử lý: {processed_count} ảnh...")
            else:
                skipped_count += 1
                
print("-" * 30)
print(f"HOÀN THÀNH!")
print(f"Số ảnh đã crop thành công: {processed_count}")
print(f"Số ảnh bị bỏ qua (không thấy tay): {skipped_count}")
print(f"Kiểm tra folder mới tại: {OUTPUT_DIR}")