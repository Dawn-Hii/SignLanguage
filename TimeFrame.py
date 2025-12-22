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
my_video_2 = r"C:\Users\Predator\Videos\Video_sign_language\601477544_25515138254805696_7524847384678863106_n.mp4"

# Thời gian của video 
my_segments_2 = [
    ("A",2,5),("B",9, 16),("C",18,25),("D",30,32),("E",41,43),("G",49,50),("H",58,60)     
    ,("I",64,66),("K",73,75),("L",79,81),("M",88,90),("N",94,96),("O",102,104),("P",111,113)     
    ,("Q",121,123),("R",130,132),("S",138,139),("T",149,150),("U",156,157),("V",161,163),("X",175,176),("Y",182,184)     
]

extract_frames_from_video(
    video_path=my_video_2, 
    output_root="dataset_sign_language",  
    segments=my_segments_2,
    frame_skip=5 
)