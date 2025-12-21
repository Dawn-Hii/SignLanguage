import os

def reorder_dataset(dataset_root):
    # Kiểm tra đường dẫn
    if not os.path.exists(dataset_root):
        print(f"Lỗi: Không tìm thấy thư mục {dataset_root}")
        return

    # Lấy danh sách các folder con (A, B, C...)
    folders = sorted(os.listdir(dataset_root))
    
    for folder_name in folders:
        folder_path = os.path.join(dataset_root, folder_name)
        
        # Chỉ xử lý nếu là thư mục
        if not os.path.isdir(folder_path):
            continue
            
        # Lấy danh sách file ảnh .jpg hiện có
        files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        
        # Sắp xếp file theo tên cũ để giữ đúng thứ tự thời gian
        # (A_001, A_005, A_010...)
        files.sort()
        
        if not files:
            continue
            
        print(f"Đang xử lý folder [{folder_name}]: {len(files)} ảnh...")

        # --- BƯỚC 1: Đổi sang tên tạm (temp) ---
        # Mục đích: Tránh lỗi "File already exists" nếu đổi trực tiếp A_003 thành A_002
        for i, filename in enumerate(files):
            old_path = os.path.join(folder_path, filename)
            # Tạo tên tạm: temp_0.tmp, temp_1.tmp...
            temp_path = os.path.join(folder_path, f"temp_{i}.tmp")
            os.rename(old_path, temp_path)

        # --- BƯỚC 2: Đổi sang tên chuẩn mới ---
        # Lấy danh sách file tạm
        temp_files = [f for f in os.listdir(folder_path) if f.endswith('.tmp')]
        
        # Sắp xếp theo số thứ tự trong tên tạm (temp_1, temp_2...)
        # Bước này cực quan trọng để ảnh không bị xáo trộn thứ tự
        temp_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        for i, filename in enumerate(temp_files):
            old_path = os.path.join(folder_path, filename)
            
            # Tạo tên mới: FolderName_001.jpg (số bắt đầu từ 1)
            new_name = f"{folder_name}_{i+1:03d}.jpg" 
            new_path = os.path.join(folder_path, new_name)
            
            os.rename(old_path, new_path)

    print("-" * 30)
    print("HOÀN THÀNH! Các file đã được sắp xếp lại theo thứ tự 001 -> hết.")

# --- CẤU HÌNH ---
# Đường dẫn đến thư mục chứa data của bạn
# Nhớ thêm chữ r đằng trước
my_dataset_path = r"D:\Train\AI\SignLanguage\dataset_sign_language"

# Chạy hàm
reorder_dataset(my_dataset_path)