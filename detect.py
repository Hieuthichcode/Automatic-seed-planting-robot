import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import tflite_runtime.interpreter as tflite
import os

# ===================== CONFIG =====================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model", "640fit-fp16.tflite")
INPUT_SIZE = 640
NUM_THREADS = 4
CONF_THRESH = 0.4
NMS_THRESH = 0.45
CLASS_FILE = os.path.join(APP_DIR, "classes.txt")  # nếu không có thì fallback
SKIP_FRAMES = 2  # Process mỗi 2 frame (giảm để tăng FPS hiển thị)

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
print(f"Loading TFLite model: {MODEL_PATH}")
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded. Input details:", input_details)
print("Output details:", output_details)

# Detect output shape to know format
out_shape = output_details[0]['shape']  # e.g. [1,25200,85]
print("Output shape:", out_shape)

# ===================== HELPERS =====================
def letterbox(img, new_shape=(INPUT_SIZE, INPUT_SIZE), color=(114,114,114)):
    """Resize and pad image while meeting stride-multiple constraints (like YOLOv5 letterbox).
       Returns resized image, scale, and padding (dw, dh).
    """
    shape = img.shape[:2]  # current shape (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2
    dh /= 2

    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.0001)), int(round(dh + 0.0001))
    left, right = int(round(dw - 0.0001)), int(round(dw + 0.0001))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (left, top)

def scale_coords(box_xywh, r, pad, orig_shape):
    """Scale boxes from letterboxed image back to original image size.
       box_xywh: [x_center, y_center, w, h] in pixels relative to input_size
       r: scale ratio used in letterbox
       pad: (pad_x, pad_y)
       orig_shape: (h, w)
    """
    pad_x, pad_y = pad
    # remove padding
    x_center = (box_xywh[0] - pad_x) / r
    y_center = (box_xywh[1] - pad_y) / r
    w = box_xywh[2] / r
    h = box_xywh[3] / r
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)
    # clamp
    x1 = max(0, min(orig_shape[1]-1, x1))
    y1 = max(0, min(orig_shape[0]-1, y1))
    x2 = max(0, min(orig_shape[1]-1, x2))
    y2 = max(0, min(orig_shape[0]-1, y2))
    return [x1, y1, x2, y2]

def nms_boxes(boxes, scores, iou_threshold=NMS_THRESH):
    """Use OpenCV NMSBoxes. boxes in [x,y,w,h] format, scores list"""
    if len(boxes) == 0:
        return []
    rects = [[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in boxes]  # convert to x,y,w,h
    indices = cv2.dnn.NMSBoxes(rects, scores, CONF_THRESH, iou_threshold)
    if len(indices) == 0:
        return []
    # cv2 returns list of [[i], [j], ...] or a tuple
    indices = np.array(indices).reshape(-1)
    return indices.tolist()

# ===================== INFERENCE WORKER =====================
def infer_on_image(img_bgr):
    """Preprocess with letterbox, run tflite interpreter, return raw model output + r, pad"""
    start_time = time.time()
    
    img_padded, r, pad = letterbox(img_bgr, (INPUT_SIZE, INPUT_SIZE))
    input_tensor = img_padded.astype(np.float32) / 255.0
    # If model expects NHWC float32
    input_index = input_details[0]['index']
    # If model expects different dtype, cast:
    if input_details[0]['dtype'] == np.float32:
        interpreter.set_tensor(input_index, np.expand_dims(input_tensor, 0).astype(np.float32))
    else:
        # handle uint8 quantized models if needed (rare for your fp16)
        interpreter.set_tensor(input_index, np.expand_dims(input_tensor, 0).astype(input_details[0]['dtype']))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    inference_time = time.time() - start_time
    # output_data shape is [1, N, C] -> we return [N, C]
    return output_data[0], r, pad, inference_time

# ===================== POSTPROCESS =====================
def postprocess_and_draw(output, r, pad, frame):
    """Decode model output (assuming [x,y,w,h,obj_conf,class_scores...])"""
    h0, w0 = frame.shape[:2]
    boxes = []
    scores = []
    class_ids = []

    # output is (N, 5 + num_classes)
    for det in output:
        # typical det: [x_center_norm?? or px?, y_center, w, h, obj_conf, class0, class1, ...]
        # Many exported tflite outputs are in normalized [0-1] if export used relative coords,
        # but many are in pixel coords relative to INPUT_SIZE. We will detect type by value range.
        # If x,y,w,h <= 1 -> normalized, else pixel coords.
        # Here we assume values are relative to INPUT_SIZE (pixel) in many exports; handle both.
        x, y, w, hh, obj_conf = det[0], det[1], det[2], det[3], det[4]
        class_confs = det[5:]
        class_id = int(np.argmax(class_confs))
        class_conf = float(class_confs[class_id])
        final_conf = float(obj_conf) * class_conf
        if final_conf < CONF_THRESH:
            continue

        # Decide if coords normalized (0..1) or pixel (0..INPUT_SIZE)
        # Check by typical range: if max(x,y,w,hh) <= 1.01 treat as normalized
        if max(x, y, w, hh) <= 1.01:
            # normalized -> scale to INPUT_SIZE
            x_px = x * INPUT_SIZE
            y_px = y * INPUT_SIZE
            w_px = w * INPUT_SIZE
            h_px = hh * INPUT_SIZE
        else:
            x_px = x
            y_px = y
            w_px = w
            h_px = hh

        # box in letterboxed image coords as x_center, y_center, w, h
        box_xywh = [x_px, y_px, w_px, h_px]
        # scale back to original image
        x1, y1, x2, y2 = scale_coords(box_xywh, r, pad, (h0, w0))
        boxes.append([x1, y1, x2, y2])
        scores.append(final_conf)
        class_ids.append(class_id)

    # Apply NMS
    indices = nms_boxes(boxes, scores, iou_threshold=NMS_THRESH)
    if len(indices) == 0:
        return frame

    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        cls = class_ids[i]
        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class{cls}"
        conf = scores[i]
        
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

    return frame

# ===================== CAMERA + MULTI-THREAD LOOP =====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# ===== TỐI ƯU: Giảm buffer size để giảm độ trễ =====
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Chỉ giữ 1 frame trong buffer
print("Camera buffer size set to 1")

executor = ThreadPoolExecutor(max_workers=2)
future = None
latest_display = None

# ===== TỐI ƯU: FPS Counter chính xác =====
frame_count = 0
skip_counter = 0
fps_display = 0.0
fps_start_time = time.time()
inference_times = []

print("Starting main loop. Press ESC to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        break

    frame_count += 1
    skip_counter += 1

    # If previous inference finished -> fetch result and draw on the current frame
    if future is not None and future.done():
        try:
            output, r, pad, inf_time = future.result()
            # draw using the current frame (we should use a fresh frame copy to avoid mismatch)
            latest_display = postprocess_and_draw(output, r, pad, frame.copy())
            inference_times.append(inf_time)
            # Giữ tối đa 30 giá trị để tính trung bình
            if len(inference_times) > 30:
                inference_times.pop(0)
        except Exception as e:
            print("Error processing future result:", e)
            latest_display = frame

    # ===== TỐI ƯU: Skip Frame - chỉ process mỗi SKIP_FRAMES frame =====
    if (future is None or future.done()) and skip_counter >= SKIP_FRAMES:
        skip_counter = 0
        future = executor.submit(infer_on_image, frame.copy())

    # ===== TỐI ƯU: FPS Counter chính xác =====
    current_time = time.time()
    elapsed = current_time - fps_start_time
    if elapsed >= 1.0:  # Cập nhật FPS mỗi giây
        fps_display = frame_count / elapsed
        frame_count = 0
        fps_start_time = current_time

    # Tính average inference time
    avg_inf_time = sum(inference_times) / len(inference_times) if inference_times else 0

    # Show the latest display (if available) or raw frame
    disp = latest_display if latest_display is not None else frame
    cv2.putText(disp, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(disp, f"Inference: {avg_inf_time*1000:.0f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow("Plant Disease Detection (Optimized)", disp)

    # Update fps_display approx from model timing if available
    # We cannot get exact per-frame model time from this thread (unless returned). Use simple moving avg:
    # optional: could compute model time inside infer_on_image and return it; skipped for brevity.

    if cv2.waitKey(1) & 0xFF == 27:
        break

# cleanup
executor.shutdown(wait=True)
cap.release()
cv2.destroyAllWindows()
print("Exiting.")
