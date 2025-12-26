import os
import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp


@dataclass
class Config:
    # Tự động thử camera 0 trước, nếu không được thì thử 1
    out_root: str = "dataset_bw_new"
    image_size: int = 224
    jpg_quality: int = 95

    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7

    # --- CẤU HÌNH CHỤP ---
    min_capture_interval_s: float = 0.5  # Tốc độ chụp

    min_bbox_size_px: int = 50
    max_bbox_size_ratio: float = 0.9
    bbox_margin_ratio: float = 0.35

    # --- CHỐNG GIẬT ---
    bbox_smooth_alpha: float = 0.6 
    
    # --- HIỆU ỨNG ---
    flash_frames: int = 3 


CFG = Config()


class BBoxSmoother:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.prev_bbox: Optional[List[float]] = None 

    def update(self, new_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        nx1, ny1, nx2, ny2 = new_bbox
        
        if self.prev_bbox is None:
            self.prev_bbox = [float(nx1), float(ny1), float(nx2), float(ny2)]
            return new_bbox
        
        px1, py1, px2, py2 = self.prev_bbox
        
        # Công thức làm mượt chuyển động (Exponential Moving Average)
        sx1 = self.alpha * nx1 + (1 - self.alpha) * px1
        sy1 = self.alpha * ny1 + (1 - self.alpha) * py1
        sx2 = self.alpha * nx2 + (1 - self.alpha) * px2
        sy2 = self.alpha * ny2 + (1 - self.alpha) * py2
        
        self.prev_bbox = [sx1, sy1, sx2, sy2]
        return int(sx1), int(sy1), int(sx2), int(sy2)

    def reset(self):
        self.prev_bbox = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_label(label: str) -> str:
    label = label.strip()
    if not label: return "Unknown"
    return re.sub(r"[^\w\-]+", "_", label)


def next_index_in_folder(folder: str, label: str) -> int:
    if not os.path.isdir(folder): return 1
    # Tìm file cuối cùng có dạng: Label_Số.jpg
    pat = re.compile(rf"^{re.escape(label)}_(\d+)\.jpg$", re.IGNORECASE)
    mx = 0
    for fn in os.listdir(folder):
        m = pat.match(fn)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def pad_to_square(img: np.ndarray, pad_value: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w: return img
    
    # Chèn viền đen để ảnh thành hình vuông 
    if h > w:
        diff = h - w
        left = diff // 2
        right = diff - left
        return cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=pad_value)
    else:
        diff = w - h
        top = diff // 2
        bottom = diff - top
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=pad_value)


def landmarks_to_points(landmarks, w: int, h: int) -> np.ndarray:
    return np.array([
        [int(np.clip(lm.x * w, 0, w - 1)), int(np.clip(lm.y * h, 0, h - 1))]
        for lm in landmarks.landmark
    ], dtype=np.int32)


def compute_bbox_from_points(pts: np.ndarray, img_w: int, img_h: int, margin_ratio: float) -> Tuple[int, int, int, int]:
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    
    bw, bh = x2 - x1, y2 - y1
    margin = int(max(bw, bh) * margin_ratio)
    
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img_w - 1, x2 + margin)
    y2 = min(img_h - 1, y2 + margin)
    
    return x1, y1, x2, y2


def crop_image_rect(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return frame[y1:y2+1, x1:x2+1].copy()


def make_preview(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int], 
                 auto_mode: bool, status: str, total_saved: int, 
                 label: str, extra: str = "") -> np.ndarray:
    vis = frame_bgr.copy()
    x1, y1, x2, y2 = bbox
    
    # Màu khung: Xanh lá (Lưu thành công) | Vàng (Chờ) | Đỏ (Lỗi/Biên)
    color = (0, 255, 0) if "SAVED" in status else (0, 255, 255)
    if "TOO" in status or "BORDER" in status:
        color = (0, 0, 255)
        
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    
    # Hiển thị thông tin
    infos = [
        f"LABEL: {label}",
        f"AUTO: {'ON' if auto_mode else 'OFF'} (Press 's')",
        f"SAVED: {total_saved}",
        f"STATUS: {status}",
    ]
    if extra: infos.append(extra)

    y = 30
    for line in infos:
        # Vẽ viền chữ đen cho dễ đọc
        cv2.putText(vis, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        # Vẽ chữ trắng
        cv2.putText(vis, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y += 25
        
    cv2.putText(vis, "[Q] Quit", (15, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return vis


def main() -> None:
    # 1. Nhập nhãn
    label_input = input("Nhập tên Chữ cái/Nhãn (ví dụ: A): ")
    try:
        label = sanitize_label(label_input)
    except ValueError:
        print("Tên nhãn không hợp lệ!")
        return

    out_dir = os.path.join(CFG.out_root, label)
    ensure_dir(out_dir)
    idx = next_index_in_folder(out_dir, label)
    
    print(f"Lưu ảnh vào: {out_dir}")
    print(f"Bắt đầu từ số: {idx}")

    # 2. Mở Camera 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được Camera 0. Đang thử Camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            raise RuntimeError("Không tìm thấy camera nào!")

    # 3. Setup MediaPipe
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=CFG.max_num_hands,
        min_detection_confidence=CFG.min_detection_confidence,
        min_tracking_confidence=CFG.min_tracking_confidence,
        model_complexity=1,
    )
    
    smoother = BBoxSmoother(alpha=CFG.bbox_smooth_alpha)
    auto_mode = False
    last_capture_t = 0.0
    flash_counter = 0

    print("Sẵn sàng! Nhấn 's' để Tự động chụp, 'q' để Thoát.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None: continue

        # Lật ảnh để soi gương ---
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(frame_rgb)
        
        vis = frame.copy()
        status = "READY"
        
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            h, w = frame.shape[:2]
            pts = landmarks_to_points(lm, w, h)
            
            # Tính toán BBox
            raw_bbox = compute_bbox_from_points(pts, w, h, CFG.bbox_margin_ratio)
            bbox = smoother.update(raw_bbox)
            
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            
            valid_hand = True

            # Kiểm tra điều kiện (Quá xa, quá gần, chạm viền)
            if bw < CFG.min_bbox_size_px or bh < CFG.min_bbox_size_px:
                status = "TOO_FAR"
                valid_hand = False
            elif max(bw, bh) > int(min(h, w) * CFG.max_bbox_size_ratio):
                status = "TOO_CLOSE"
                valid_hand = False
            elif x1 <= 1 or y1 <= 1 or x2 >= w - 2 or y2 >= h - 2:
                status = "BORDER_TOUCH"
                valid_hand = False
            
            # Cắt ảnh & Xử lý (Về Grayscale -> Resize 224x224)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_crop = crop_image_rect(gray_frame, bbox)
            roi_square = pad_to_square(roi_crop, pad_value=0)
            final_img = cv2.resize(roi_square, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_AREA)

            # Vẽ preview nhỏ ở góc
            preview_size = 150
            preview_img = cv2.resize(final_img, (preview_size, preview_size), interpolation=cv2.INTER_NEAREST)
            preview_img_bgr = cv2.cvtColor(preview_img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(preview_img_bgr, (0,0), (preview_size-1, preview_size-1), (0,255,0), 2)
            
            # Dán preview vào góc phải trên (lưu ý tọa độ sau khi flip)
            if w > preview_size and h > preview_size:
                vis[10:10+preview_size, w-10-preview_size:w-10] = preview_img_bgr

            # LOGIC CHỤP
            now = time.time()
            if auto_mode and valid_hand:
                if (now - last_capture_t) >= CFG.min_capture_interval_s:
                    fn = f"{label}_{idx}.jpg"
                    out_path = os.path.join(out_dir, fn)
                    
                    # Lưu ảnh chất lượng cao
                    cv2.imwrite(out_path, final_img, [int(cv2.IMWRITE_JPEG_QUALITY), CFG.jpg_quality])
                    
                    idx += 1
                    last_capture_t = now
                    status = "SAVED"
                    flash_counter = CFG.flash_frames

            vis = make_preview(vis, bbox, auto_mode, status, idx - 1, label)
        else:
            smoother.reset()
            # Hiển thị thông báo khi không thấy tay
            cv2.putText(vis, "NO HAND DETECTED", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(vis, f"SAVED: {idx - 1}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Hiệu ứng nháy Flash màn hình khi chụp
        if flash_counter > 0:
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (vis.shape[1], vis.shape[0]), (255, 255, 255), -1)
            vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)
            flash_counter -= 1

        cv2.imshow("Hand Data Collector (Mirrored)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        if key == ord("s"):
            auto_mode = not auto_mode
            last_capture_t = time.time() - CFG.min_capture_interval_s
            print(f"Auto mode: {auto_mode}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()