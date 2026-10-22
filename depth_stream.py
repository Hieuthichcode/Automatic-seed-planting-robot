"""
depth_stream.py
Module điều khiển luồng camera chiều sâu (depth) của Intel RealSense D435.

- Chỉ khởi động pipeline khi có request, tắt khi không dùng để tiết kiệm tài nguyên.
- Depth frame được colormap (JET) và encode JPEG trước khi stream.
- Fallback: nếu không có pyrealsense2 thì trả về frame thông báo lỗi.
"""

import cv2
import numpy as np
import threading
import time

try:
    import pyrealsense2 as rs
    _HAS_REALSENSE = True
except ImportError:
    _HAS_REALSENSE = False

# ── State ──────────────────────────────────────────────────────────────────
_lock        = threading.Lock()
_pipeline    = None
_align       = None          # rs.align để căn depth cùng góc nhìn với color
_running     = False
_last_fps         = 0.0      # FPS mới nhất (để /depth_status đọc)
_last_center_dist = 0.0      # Khoảng cách điểm giữa mới nhất (m)
_colormap    = cv2.COLORMAP_JET

# Cấu hình depth
DEPTH_WIDTH  = 1280
DEPTH_HEIGHT = 720
DEPTH_FPS    = 30
DEPTH_MAX_M  = 4.0           # Khoảng cách tối đa hiển thị (mét)


def start_depth() -> bool:
    """
    Khởi động depth pipeline.
    Trả về True nếu thành công (hoặc đã chạy rồi).
    """
    global _pipeline, _align, _running

    with _lock:
        if _running:
            return True
        if not _HAS_REALSENSE:
            print("[Depth] pyrealsense2 không có – không thể khởi động")
            return False
        try:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT,
                              rs.format.z16, DEPTH_FPS)
            # Bật thêm color để align (nếu cần overlay sau này)
            cfg.enable_stream(rs.stream.color, DEPTH_WIDTH, DEPTH_HEIGHT,
                              rs.format.bgr8, DEPTH_FPS)

            _pipeline = rs.pipeline()
            profile = _pipeline.start(cfg)

            # Căn chỉnh depth sang không gian color
            _align = rs.align(rs.stream.color)

            # Bật High Accuracy preset nếu thiết bị hỗ trợ
            try:
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
            except Exception:
                pass

            _running = True
            print(f"[Depth] Pipeline started ({DEPTH_WIDTH}x{DEPTH_HEIGHT}@{DEPTH_FPS}fps)")
            return True
        except Exception as e:
            print(f"[Depth] Failed to start pipeline: {e}")
            _pipeline = None
            _running  = False
            return False


def stop_depth():
    """Dừng depth pipeline và giải phóng tài nguyên."""
    global _pipeline, _align, _running

    with _lock:
        if not _running:
            return
        try:
            _pipeline.stop()
        except Exception:
            pass
        _pipeline = None
        _align    = None
        _running  = False
        print("[Depth] Pipeline stopped")


def is_running() -> bool:
    return _running


def _depth_to_colormap(depth_frame) -> np.ndarray:
    """
    Chuyển depth frame (z16) sang ảnh BGR colormap.
    Clamp giá trị theo DEPTH_MAX_M để loại bỏ nhiễu xa.
    """
    depth_image = np.asanyarray(depth_frame.get_data())   # mm units, uint16

    # Clamp: giữ giá trị trong [0, max_dist_mm]
    max_dist_mm = int(DEPTH_MAX_M * 1000)
    depth_clipped = np.clip(depth_image, 0, max_dist_mm).astype(np.float32)

    # Normalize về [0, 255]
    depth_norm = (depth_clipped / max_dist_mm * 255).astype(np.uint8)

    # Áp colormap
    colorized = cv2.applyColorMap(depth_norm, _colormap)
    return colorized


def generate_depth_frames():
    """
    Generator stream MJPEG cho depth camera.
    Tự động start pipeline khi gọi lần đầu.
    """
    global _last_fps, _last_center_dist
    if not start_depth():
        # Không có RealSense – trả frame thông báo lỗi
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "RealSense D435 not available",
                    (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        ret, buf = cv2.imencode('.jpg', error_img)
        if ret:
            data = buf.tobytes()
            while True:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
                time.sleep(1.0)
        return

    frame_count = 0
    fps_start   = time.time()
    fps_val     = 0.0

    while _running:
        try:
            frames = _pipeline.wait_for_frames(timeout_ms=500)

            # Align depth → color space
            aligned = _align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame:
                continue

            # Chuyển depth → colormap BGR
            depth_colormap = _depth_to_colormap(depth_frame)

            # Overlay thông tin
            frame_count += 1
            now = time.time()
            elapsed = now - fps_start
            if elapsed >= 1.0:
                fps_val     = frame_count / elapsed
                _last_fps   = fps_val
                frame_count = 0
                fps_start   = now

            # Lấy khoảng cách điểm giữa ảnh
            cx = DEPTH_WIDTH  // 2
            cy = DEPTH_HEIGHT // 2
            center_dist = depth_frame.get_distance(cx, cy)
            _last_center_dist = center_dist
            dist_str = f"{center_dist:.2f}m" if center_dist > 0 else "N/A"

            cv2.putText(depth_colormap, f"FPS: {fps_val:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_colormap, f"Center: {dist_str}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(depth_colormap, f"Max: {DEPTH_MAX_M:.1f}m",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Dấu chữ thập giữa
            cv2.drawMarker(depth_colormap, (cx, cy), (255, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)

            # Encode JPEG
            ret, buf = cv2.imencode('.jpg', depth_colormap,
                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

            time.sleep(0.01)

        except Exception as e:
            print(f"[Depth] Stream error: {e}")
            time.sleep(0.2)
