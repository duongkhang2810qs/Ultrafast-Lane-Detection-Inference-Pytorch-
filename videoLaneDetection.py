import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

import cv2
import numpy as np
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# ====== Cấu hình ======
model_path = "models/tusimple_18.pth"   # hoặc "models/culane_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False

input_path = "input1.mp4"                # đổi tên cho khớp file của bạn
output_path = "output1.avi"              # xuất AVI

def to_bgr_u8(img):
    if img is None:
        raise RuntimeError("Output frame is None")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

# ====== Mở video vào ======
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"Không mở được video: {input_path}")

# Đọc frame đầu
ret, frame = cap.read()
if not ret or frame is None:
    cap.release()
    raise RuntimeError("Không đọc được frame đầu tiên.")

# ====== Model ======
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

# Detect trước 1 frame để chốt kích thước writer
first_out = lane_detector.detect_lanes(frame)
first_out = to_bgr_u8(first_out)

# Bắt buộc kích thước chẵn cho một số codec
oh, ow = first_out.shape[:2]
target_w = ow - (ow % 2)
target_h = oh - (oh % 2)
if (target_w, target_h) != (ow, oh):
    first_out = cv2.resize(first_out, (target_w, target_h), interpolation=cv2.INTER_AREA)

# FPS fallback nếu không đọc được từ file
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

# ====== VideoWriter (AVI/MJPG) ======
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
if not writer.isOpened():
    cap.release()
    raise RuntimeError("Không tạo được VideoWriter (AVI/MJPG). Thử đổi file đầu vào hoặc codec khác.")

# Ghi frame đầu
writer.write(first_out)

# Vòng lặp các frame còn lại
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    try:
        out_img = lane_detector.detect_lanes(frame)
        out_img = to_bgr_u8(out_img)
        # Đảm bảo kích thước khớp writer
        if out_img.shape[1] != target_w or out_img.shape[0] != target_h:
            out_img = cv2.resize(out_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        writer.write(out_img)
    except Exception as e:
        # Bỏ qua frame lỗi nhưng tiếp tục video
        print("Skip 1 frame:", e)
        continue

cap.release()
writer.release()
print(f"Đã lưu video kết quả: {output_path}")
