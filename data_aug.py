import cv2
import numpy as np
import os
import random
import glob

INPUT_FOLDER = "dataset_cropped"      # Ảnh gốc 
OUTPUT_FOLDER = "aug3"       # Thư mục kết quả 
NUM_AUGMENT = 5                       # 1 ảnh gốc : 5 ảnh mới


def strong_augment(image):
    """Hàm biến đổi ảnh: Xoay, Zoom, Nhiễu, Sáng"""
    h, w = image.shape[:2]
    
    # 1. Xoay & Zoom an toàn (-20 đến 20 độ, Zoom 0.9-1.1)
    angle = random.uniform(-20, 20)
    scale = random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
    
    # Dịch chuyển nhẹ (10%)
    M[0, 2] += random.uniform(-0.1, 0.1) * w
    M[1, 2] += random.uniform(-0.1, 0.1) * h
    
    # Border màu đen để AI không học nhiễu viền
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # 2. Chỉnh sáng/tương phản
    if random.random() < 0.7:
        image = cv2.convertScaleAbs(image, alpha=random.uniform(0.8, 1.2), beta=random.uniform(-40, 40))

    # 3. Thêm nhiễu (Giúp AI học tốt webcam mờ)
    if random.random() < 0.5:
        noise = np.random.normal(0, random.randint(10, 40), image.shape).astype('int16')
        image = cv2.add(image.astype('int16'), noise)
        image = np.clip(image, 0, 255).astype('uint8')

    # 4. Làm mờ nhẹ
    if random.random() < 0.3:
        k = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (k, k), 0)

    return image

def process_and_rename():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Lỗi: Không thấy thư mục '{INPUT_FOLDER}'")
        return

    print(f"BẮT ĐẦU XỬ LÝ: {INPUT_FOLDER} -> {OUTPUT_FOLDER}")

    # Duyệt qua từng lớp (A, B, C...)
    for class_name in os.listdir(INPUT_FOLDER):
        src_path = os.path.join(INPUT_FOLDER, class_name)
        dst_path = os.path.join(OUTPUT_FOLDER, class_name)
        
        if not os.path.isdir(src_path): continue
        os.makedirs(dst_path, exist_ok=True)

        # Lấy danh sách ảnh gốc
        files = glob.glob(os.path.join(src_path, "*"))
        print(f"\nClass '{class_name}': {len(files)} ảnh gốc -> Đang Augment...")

        # AUGMENTATION
        temp_count = 0
        for img_path in files:
            try:
                original = cv2.imread(img_path)
                if original is None: continue

                # Lưu ảnh gốc trước (đặt tên tạm)
                cv2.imwrite(os.path.join(dst_path, f"temp_{temp_count}_org.jpg"), original)
                temp_count += 1

                # Tạo ảnh mới
                for _ in range(NUM_AUGMENT):
                    aug_img = strong_augment(original.copy())
                    cv2.imwrite(os.path.join(dst_path, f"temp_{temp_count}_aug.jpg"), aug_img)
                    temp_count += 1
            except: pass

        # RENAME
        print(f"   Wait... Đang đổi tên chuẩn cho lớp {class_name}...")
        
        # Lấy tất cả ảnh vừa tạo ra
        all_new_files = sorted(glob.glob(os.path.join(dst_path, "*")))
        
        for idx, file_path in enumerate(all_new_files):
            # Định dạng tên: TênLớp_0001.jpg (Ví dụ: A_0001.jpg)
            new_name = f"{class_name}_{idx+1:04d}.jpg"
            new_path = os.path.join(dst_path, new_name)
            
            os.rename(file_path, new_path)

        print(f"Hoàn tất: {len(all_new_files)} ảnh ")

    print(f"\nXONG TOÀN BỘ! Dữ liệu nằm trong '{OUTPUT_FOLDER}'")

# Chạy chương trình
if __name__ == "__main__":
    process_and_rename()
