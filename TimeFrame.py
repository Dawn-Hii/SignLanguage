import cv2
import os

def extract_frames_from_video(video_path, output_root, segments, frame_skip=5):
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy video tại {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video FPS: {fps}")
    print(f"Đang xử lý video: {os.path.basename(video_path)}")

    frame_count = 0
    saved_count = {} 

    while True:
        success, frame = cap.read()
        if not success:
            break 

        current_time = frame_count / fps
        
        # Tìm label hiện tại
        current_label = None
        for label, start_sec, end_sec in segments:
            if start_sec <= current_time <= end_sec:
                current_label = label
                break
        
        if current_label and (frame_count % frame_skip == 0):
            label_dir = os.path.join(output_root, current_label)
            
            # Nếu chưa đếm số lượng file trong folder này lần nào
            if current_label not in saved_count:
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)
                    saved_count[current_label] = 0
                else:
                    # Đếm xem hiện tại đang có bao nhiêu file ảnh .jpg rồi
                    existing_files = [f for f in os.listdir(label_dir) if f.endswith('.jpg')]
                    saved_count[current_label] = len(existing_files)
            # ----------------------------------
            
            saved_count[current_label] += 1
            
            file_name = f"{current_label}_{saved_count[current_label]:03d}.jpg"
            save_path = os.path.join(label_dir, file_name)
            
            cv2.imwrite(save_path, frame)
            
            if saved_count[current_label] % 50 == 0:
                print(f"[{current_label}] Đã lưu đến ảnh thứ: {saved_count[current_label]}")

        frame_count += 1

    cap.release()
    print("--- Đã thêm dữ liệu mới thành công! ---")



# Link video 
my_video_2 = r"C:\Users\Predator\Pictures\Camera Roll\WIN_20251222_22_46_08_Pro.mp4"

# Thời gian của video 
my_segments_2 = [("G",6,11), ("H",19,31)     
]

#("A",2,17),("B",23, 39),("C",45,56),("D",60,78),("E",83,94),, ("I",146,160), ("K",170,181),("L",185,200),("M",211,223),("N",228,241),("O",248,260),("P",268,290),     
#("Q",295,307),("R",316,336),("S",342,360),("T",366,386),("U",392,402),("V",407,418),("X",426,440),("Y",448,465)
extract_frames_from_video(
    video_path=my_video_2, 
    output_root="dataset_sign_language",  
    segments=my_segments_2,
    frame_skip=5
)