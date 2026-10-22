import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import os
import threading

# Try import RealSense SDK
try:
    import pyrealsense2 as rs
    _HAS_REALSENSE = True
    print("[INFO] pyrealsense2 found – using RealSense D435")
except ImportError:
    _HAS_REALSENSE = False
    print("[WARNING] pyrealsense2 not found – using the configured OpenCV camera")

# Try import TFLite runtime, fallback to TensorFlow if not available
try:
    import tflite_runtime.interpreter as tflite
    print("[INFO] Using tflite_runtime")
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        print("[INFO] Using TensorFlow Lite from tensorflow package")
    except ImportError:
        print("[WARNING] Neither tflite_runtime nor tensorflow found!")
        print("[WARNING] Detection features will be disabled")
        tflite = None

# ===================== CONFIG =====================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
MODEL_NAME = "new_float32.tflite"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
INPUT_SIZE = 640
NUM_THREADS = 4
CONF_THRESH = 0.3
NMS_THRESH = 0.4
CLASS_FILE = os.path.join(APP_DIR, "classes.txt")
SKIP_FRAMES = 2
OPENCV_CAMERA_DEVICE = os.environ.get("ROBOT_CAMERA_DEVICE", "0").strip()

# ===================== ENV TWEAKS =====================
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)

# ===================== LOAD CLASSES =====================
if os.path.exists(CLASS_FILE):
    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        CLASS_NAMES = [c.strip() for c in f.readlines() if c.strip()]
else:
    # Fallback cho 2 classes: 0=Bệnh, 1=Bình thường
    CLASS_NAMES = ["Benh", "Binh_thuong"]

# ===================== LOAD TFLITE MODEL =====================
interpreter = None
input_details = None
output_details = None
MODEL_INPUT_HEIGHT = INPUT_SIZE
MODEL_INPUT_WIDTH = INPUT_SIZE
MODEL_OUTPUT_COLUMNS = len(CLASS_NAMES) + 5
MODEL_OUTPUT_HAS_OBJECTNESS = True

if tflite is not None and os.path.exists(MODEL_PATH):
    try:
        print(f"Loading TFLite model: {MODEL_PATH}")
        interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        model_input_shape = input_details[0]["shape"]
        if len(model_input_shape) != 4 or int(model_input_shape[-1]) != 3:
            raise ValueError(
                f"Unsupported model input shape: {model_input_shape}"
            )
        MODEL_INPUT_HEIGHT = int(model_input_shape[1])
        MODEL_INPUT_WIDTH = int(model_input_shape[2])
        if MODEL_INPUT_HEIGHT <= 0 or MODEL_INPUT_WIDTH <= 0:
            raise ValueError(
                f"Dynamic model input shape is not supported: {model_input_shape}"
            )

        model_output_shape = tuple(
            int(value) for value in output_details[0]["shape"]
        )
        output_dimensions = {
            value for value in model_output_shape if value > 1
        }
        columns_with_objectness = len(CLASS_NAMES) + 5
        columns_without_objectness = len(CLASS_NAMES) + 4
        if columns_with_objectness in output_dimensions:
            MODEL_OUTPUT_COLUMNS = columns_with_objectness
            MODEL_OUTPUT_HAS_OBJECTNESS = True
        elif columns_without_objectness in output_dimensions:
            MODEL_OUTPUT_COLUMNS = columns_without_objectness
            MODEL_OUTPUT_HAS_OBJECTNESS = False
        else:
            raise ValueError(
                f"Unsupported model output shape {model_output_shape} for "
                f"{len(CLASS_NAMES)} classes"
            )
        print(
            "Model loaded successfully. Input: "
            f"{MODEL_INPUT_WIDTH}x{MODEL_INPUT_HEIGHT}; output: "
            f"{model_output_shape}; objectness: "
            f"{'yes' if MODEL_OUTPUT_HAS_OBJECTNESS else 'no'}"
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        interpreter = None
elif tflite is None:
    print("[WARNING] TFLite not available - detection will be disabled")
else:
    print(f"[WARNING] Model file not found: {MODEL_PATH}")

# ===================== GLOBAL VARIABLES =====================
# RealSense pipeline (RGB)
_rs_pipeline = None
_rs_config   = None
cap = None          # OpenCV fallback
detection_enabled = False
latest_frame = None
frame_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=2)
future = None
latest_display = None
fps_display = 0.0
avg_inf_time = 0.0
object_count = 0
disease_count = 0  # Số cây bị bệnh
healthy_count = 0  # Số cây bình thường
interpreter_lock = threading.Lock()
threshold_lock = threading.Lock()


def set_detection_thresholds(confidence, iou):
    """Update thresholds shared by realtime camera and uploaded-image modes."""
    global CONF_THRESH, NMS_THRESH
    confidence = float(confidence)
    iou = float(iou)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Ngưỡng phát hiện phải nằm trong khoảng 0 đến 1")
    if not 0.0 <= iou <= 1.0:
        raise ValueError("Ngưỡng IoU phải nằm trong khoảng 0 đến 1")
    with threshold_lock:
        CONF_THRESH = confidence
        NMS_THRESH = iou


def get_detection_thresholds():
    with threshold_lock:
        return CONF_THRESH, NMS_THRESH

# ===================== HELPERS =====================
def letterbox(img, new_shape=(INPUT_SIZE, INPUT_SIZE), color=(114,114,114)):
    """Resize and pad image while meeting stride-multiple constraints"""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.0001)), int(round(dh + 0.0001))
    left, right = int(round(dw - 0.0001)), int(round(dw + 0.0001))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (left, top)

def scale_coords(box_xywh, r, pad, orig_shape):
    """Scale boxes from letterboxed image back to original"""
    pad_x, pad_y = pad
    x_center = (box_xywh[0] - pad_x) / r
    y_center = (box_xywh[1] - pad_y) / r
    w = box_xywh[2] / r
    h = box_xywh[3] / r
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)
    x1 = max(0, min(orig_shape[1]-1, x1))
    y1 = max(0, min(orig_shape[0]-1, y1))
    x2 = max(0, min(orig_shape[1]-1, x2))
    y2 = max(0, min(orig_shape[0]-1, y2))
    return [x1, y1, x2, y2]

def nms_boxes(boxes, scores, iou_threshold=None, confidence_threshold=None):
    """NMS using OpenCV"""
    if len(boxes) == 0:
        return []
    current_confidence, current_iou = get_detection_thresholds()
    if iou_threshold is None:
        iou_threshold = current_iou
    if confidence_threshold is None:
        confidence_threshold = current_confidence
    rects = [[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in boxes]
    indices = cv2.dnn.NMSBoxes(rects, scores, confidence_threshold, iou_threshold)
    if len(indices) == 0:
        return []
    indices = np.array(indices).reshape(-1)
    return indices.tolist()

# ===================== INFERENCE WORKER =====================
def _tensor_quantization(detail):
    scale, zero_point = detail.get("quantization", (0.0, 0))
    return float(scale), int(zero_point)


def _prepare_model_input(img_padded):
    """Normalize/quantize an image according to the active TFLite model."""
    detail = input_details[0]
    dtype = np.dtype(detail["dtype"])
    # OpenCV/RealSense provide BGR, while Ultralytics TFLite models expect RGB.
    model_image = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    normalized = model_image.astype(np.float32) / 255.0
    if np.issubdtype(dtype, np.floating):
        tensor = normalized.astype(dtype)
    else:
        scale, zero_point = _tensor_quantization(detail)
        if scale > 0:
            tensor = np.rint(normalized / scale + zero_point)
        else:
            tensor = model_image
        limits = np.iinfo(dtype)
        tensor = np.clip(tensor, limits.min, limits.max).astype(dtype)
    return np.expand_dims(tensor, 0)


def _decode_model_output(raw_output):
    """Dequantize and normalize output to one row per detection."""
    detail = output_details[0]
    output = np.asarray(raw_output)
    if np.issubdtype(output.dtype, np.integer):
        scale, zero_point = _tensor_quantization(detail)
        if scale > 0:
            output = (output.astype(np.float32) - zero_point) * scale
        else:
            output = output.astype(np.float32)

    output = np.squeeze(output)
    if output.ndim != 2:
        raise ValueError(
            f"Unsupported model output shape after removing batch: {output.shape}"
        )

    if (
        output.shape[0] == MODEL_OUTPUT_COLUMNS
        and output.shape[1] != MODEL_OUTPUT_COLUMNS
    ):
        output = output.T
    if output.shape[1] != MODEL_OUTPUT_COLUMNS:
        raise ValueError(
            f"Unsupported model output layout {output.shape}; expected "
            f"{MODEL_OUTPUT_COLUMNS} values per detection"
        )
    return output


def infer_on_image(img_bgr):
    """Run inference on image"""
    # Kiểm tra nếu không có interpreter
    if interpreter is None:
        return None, 0, (0, 0), 0.0
    
    start_time = time.time()
    
    img_padded, r, pad = letterbox(
        img_bgr, (MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)
    )
    input_tensor = _prepare_model_input(img_padded)
    input_index = input_details[0]['index']
    
    # TFLite Interpreter không thread-safe. Ảnh camera và ảnh tải lên dùng
    # chung model nên mọi lần invoke phải đi qua cùng một khóa.
    with interpreter_lock:
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details[0]['index']).copy()
    
    inference_time = time.time() - start_time
    return _decode_model_output(raw_output), r, pad, inference_time

# ===================== POSTPROCESS =====================
def postprocess_and_draw_result(
    output,
    r,
    pad,
    frame,
    input_shape=None,
    output_has_objectness=None,
):
    """Decode output and return the annotated frame plus independent counts.

    ``input_shape`` is ``(height, width)``. It is optional so the production
    pipeline keeps its default size while the test tool can use each model's
    actual input resolution.
    """
    confidence_threshold, iou_threshold = get_detection_thresholds()
    if output_has_objectness is None:
        output_has_objectness = MODEL_OUTPUT_HAS_OBJECTNESS
    # Nếu không có output (interpreter = None)
    if output is None:
        return frame, {
            "plants_detected": 0,
            "disease_count": 0,
            "healthy_count": 0,
        }
    
    if input_shape is None:
        input_height = MODEL_INPUT_HEIGHT
        input_width = MODEL_INPUT_WIDTH
    elif isinstance(input_shape, int):
        input_height = input_width = int(input_shape)
    else:
        input_height, input_width = [int(value) for value in input_shape]

    h0, w0 = frame.shape[:2]
    boxes = []
    scores = []
    class_ids = []

    for det in output:
        if len(det) < 5:
            continue
        x, y, w, hh = det[0], det[1], det[2], det[3]
        if output_has_objectness:
            obj_conf = float(det[4])
            class_confs = det[5:]
        else:
            obj_conf = 1.0
            class_confs = det[4:]
        if len(class_confs) == 0:
            continue
        class_id = int(np.argmax(class_confs))
        class_conf = float(class_confs[class_id])
        final_conf = float(obj_conf) * class_conf
        if final_conf < confidence_threshold:
            continue

        if max(x, y, w, hh) <= 1.01:
            x_px = x * input_width
            y_px = y * input_height
            w_px = w * input_width
            h_px = hh * input_height
        else:
            x_px = x
            y_px = y
            w_px = w
            h_px = hh

        box_xywh = [x_px, y_px, w_px, h_px]
        x1, y1, x2, y2 = scale_coords(box_xywh, r, pad, (h0, w0))
        boxes.append([x1, y1, x2, y2])
        scores.append(final_conf)
        class_ids.append(class_id)

    indices = nms_boxes(
        boxes,
        scores,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )
    result_object_count = len(indices)

    # Đếm số lượng từng loại
    result_disease_count = 0
    result_healthy_count = 0
    
    if len(indices) == 0:
        return frame, {
            "plants_detected": 0,
            "disease_count": 0,
            "healthy_count": 0,
        }

    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        cls = class_ids[i]
        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class{cls}"
        conf = scores[i]
        
        # Đếm số lượng
        if cls == 0:
            result_disease_count += 1
        else:
            result_healthy_count += 1
        
        # Màu sắc: Đỏ nếu bệnh (class 0), Xanh lá nếu bình thường (class 1)
        if cls == 0:
            color = (0, 0, 255)  # Đỏ - Bệnh
            display_label = f"BENH {conf:.2f}"
        else:
            color = (0, 255, 0)  # Xanh lá - Bình thường
            display_label = f"BINH THUONG {conf:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        txt = display_label
        t_size = cv2.getTextSize(txt, 0, fontScale=0.7, thickness=2)[0]
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + t_size[0] + 10, y1), color, -1)
        cv2.putText(frame, txt, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    return frame, {
        "plants_detected": result_object_count,
        "disease_count": result_disease_count,
        "healthy_count": result_healthy_count,
    }


def postprocess_and_draw(output, r, pad, frame):
    """Decode camera output, draw boxes and update realtime counters."""
    global object_count, disease_count, healthy_count
    annotated, counts = postprocess_and_draw_result(output, r, pad, frame)
    object_count = counts["plants_detected"]
    disease_count = counts["disease_count"]
    healthy_count = counts["healthy_count"]
    return annotated


def process_still_image(img_bgr):
    """Run a PC-uploaded image through the exact realtime detection pipeline."""
    if interpreter is None:
        raise RuntimeError("Mô hình nhận diện chưa được tải")
    if img_bgr is None or not hasattr(img_bgr, "shape"):
        raise ValueError("Ảnh đầu vào không hợp lệ")

    output, r, pad, inference_time = infer_on_image(img_bgr)
    annotated, counts = postprocess_and_draw_result(
        output, r, pad, img_bgr.copy()
    )
    cv2.putText(
        annotated,
        f"Inference: {inference_time * 1000:.0f}ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        annotated,
        f"Cay phat hien: {counts['plants_detected']}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 100, 255),
        2,
    )
    counts["inference_time"] = round(inference_time * 1000, 1)
    confidence_threshold, iou_threshold = get_detection_thresholds()
    counts["confidence_threshold"] = confidence_threshold
    counts["nms_threshold"] = iou_threshold
    return annotated, counts

# ===================== CAMERA FUNCTIONS =====================
def init_camera():
    """Initialize RGB camera (RealSense D435 hoặc OpenCV fallback)"""
    global _rs_pipeline, _rs_config, cap

    if _HAS_REALSENSE:
        if _rs_pipeline is not None:
            return True  # Đã khởi động
        try:
            _rs_pipeline = rs.pipeline()
            _rs_config   = rs.config()
            _rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            _rs_pipeline.start(_rs_config)
            print("[RealSense] RGB pipeline started")
            return True
        except Exception as e:
            print(f"[RealSense] Failed to start RGB pipeline: {e}")
            _rs_pipeline = None
            return False
    else:
        if cap is None or not cap.isOpened():
            camera_source = (
                int(OPENCV_CAMERA_DEVICE)
                if OPENCV_CAMERA_DEVICE.isdigit()
                else OPENCV_CAMERA_DEVICE
            )
            cap = cv2.VideoCapture(camera_source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print(
                    "[Camera] OpenCV VideoCapture initialized: "
                    f"{OPENCV_CAMERA_DEVICE}"
                )
                return True
        return cap is not None and cap.isOpened()

def release_camera():
    """Release RGB camera"""
    global _rs_pipeline, cap
    if _HAS_REALSENSE and _rs_pipeline is not None:
        try:
            _rs_pipeline.stop()
        except Exception:
            pass
        _rs_pipeline = None
        print("[RealSense] RGB pipeline stopped")
    if cap is not None:
        cap.release()
        cap = None
        print("[Camera] OpenCV VideoCapture released")

def _read_rgb_frame():
    """Đọc 1 frame RGB từ nguồn đang dùng. Trả về BGR ndarray hoặc None."""
    global _rs_pipeline, cap
    if _HAS_REALSENSE and _rs_pipeline is not None:
        try:
            frames = _rs_pipeline.wait_for_frames(timeout_ms=200)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            return np.asanyarray(color_frame.get_data())  # BGR
        except Exception as e:
            print(f"[RealSense] RGB read error: {e}")
            return None
    elif cap is not None and cap.isOpened():
        ret, frame = cap.read()
        return frame if ret else None
    return None


def _infer_camera_frame(source_frame):
    """Keep inference output paired with the exact frame it belongs to."""
    output, r, pad, inference_time = infer_on_image(source_frame)
    return source_frame, output, r, pad, inference_time

def get_frame():
    """Get current frame with or without detection"""
    global latest_frame, latest_display, future, fps_display, avg_inf_time
    global frame_lock, detection_enabled

    frame = _read_rgb_frame()
    if frame is None:
        return None
    
    with frame_lock:
        latest_frame = frame.copy()
    
    if detection_enabled and interpreter is not None:
        # Detection mode - run inference
        if future is not None and future.done():
            try:
                source_frame, output, r, pad, inf_time = future.result()
                latest_display = postprocess_and_draw(
                    output, r, pad, source_frame.copy()
                )
                avg_inf_time = inf_time
            except Exception as e:
                print("Error processing detection:", e)
                latest_display = frame.copy()
        
        if future is None or future.done():
            future = executor.submit(_infer_camera_frame, frame.copy())
        
        display = latest_display.copy() if latest_display is not None else frame.copy()
        
        # Draw stats on frame
        cv2.putText(display, f"FPS: {fps_display:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(display, f"Inference: {avg_inf_time*1000:.0f}ms", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(display, f"Cay phat hien: {object_count}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,255), 2)
        
        return display
    elif detection_enabled and interpreter is None:
        # Detection requested but model not available
        cv2.putText(frame, "Detection unavailable - Model not loaded", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return frame
    else:
        # Camera only mode - no detection
        cv2.putText(frame, "Camera Mode - No Detection", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return frame

def generate_frames():
    """Generator for video streaming"""
    frame_count = 0
    fps_start_time = time.time()
    global fps_display
    
    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Update FPS
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - fps_start_time
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_start_time = current_time
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.01)  # Small delay to prevent CPU overload

def set_detection_enabled(enabled):
    """Enable or disable detection"""
    global detection_enabled, latest_display, future, object_count, disease_count, healthy_count
    detection_enabled = enabled
    if not enabled:
        latest_display = None
        future = None
        object_count = 0
        disease_count = 0
        healthy_count = 0
    print(f"Detection {'enabled' if enabled else 'disabled'}")

def get_stats():
    """Get current detection statistics"""
    confidence_threshold, iou_threshold = get_detection_thresholds()
    return {
        "fps": round(fps_display, 1),
        "plants_detected": object_count,  # Số cây phát hiện
        "disease_count": disease_count,    # Số cây bị bệnh
        "healthy_count": healthy_count,    # Số cây bình thường
        "inference_time": round(avg_inf_time * 1000, 0),
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
    }
