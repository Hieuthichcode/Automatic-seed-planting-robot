"""Tkinter tool for testing the production detection pipeline.

Input sources:
1. Intel RealSense color camera (640x480 @ 30 FPS).
2. A still image selected from the computer.

Tkinter stays on the main thread. Camera capture and model inference run on
separate worker threads. A one-item queue intentionally drops stale camera
frames so the interface stays responsive when inference is slower than 30 FPS.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)

import detect_stream  # noqa: E402  (model paths are relative to APP_DIR)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


SETTINGS_PATH = APP_DIR / "settings.json"
DEFAULT_CONFIDENCE = 0.30
DEFAULT_IOU = 0.45
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
BACKEND_TFLITE = "TensorFlow Lite"
BACKEND_YOLO = "YOLO (Ultralytics)"


def load_detection_thresholds():
    confidence = DEFAULT_CONFIDENCE
    iou = DEFAULT_IOU
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as settings_file:
            settings = json.load(settings_file)
        confidence = float(settings.get(
            "DETECTION_CONFIDENCE_THRESHOLD", confidence
        ))
        iou = float(settings.get("DETECTION_IOU_THRESHOLD", iou))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"[TEST] Không thể đọc settings.json, dùng giá trị mặc định: {exc}")

    if not 0.0 <= confidence <= 1.0:
        confidence = DEFAULT_CONFIDENCE
    if not 0.0 <= iou <= 1.0:
        iou = DEFAULT_IOU
    return confidence, iou


def draw_summary(frame, stats):
    cv2.putText(
        frame,
        f"Inference: {stats['inference_time']:.0f}ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Cay phat hien: {stats['plants_detected']}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 100, 255),
        2,
    )
    return frame


class TensorFlowLiteDetector:
    name = BACKEND_TFLITE

    def __init__(self, model_path):
        self.model_path = Path(model_path).resolve()
        self.lock = threading.Lock()

        default_model = (APP_DIR / detect_stream.MODEL_PATH).resolve()
        if (
            self.model_path == default_model
            and detect_stream.interpreter is not None
        ):
            self.interpreter = detect_stream.interpreter
            self.input_details = detect_stream.input_details
            self.output_details = detect_stream.output_details
        else:
            if detect_stream.tflite is None:
                raise RuntimeError("TensorFlow Lite chưa được cài")
            self.interpreter = detect_stream.tflite.Interpreter(
                model_path=str(self.model_path),
                num_threads=detect_stream.NUM_THREADS,
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]["shape"]
        if len(input_shape) != 4 or int(input_shape[-1]) != 3:
            raise ValueError("Model TFLite phải có input dạng NHWC với 3 kênh màu")
        self.input_height = int(input_shape[1])
        self.input_width = int(input_shape[2])
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError(
                "Model TFLite dùng kích thước input động, hiện chưa xác định "
                "được chiều cao/rộng để tiền xử lý ảnh"
            )

        output_shape = tuple(
            int(value) for value in self.output_details[0]["shape"]
        )
        self.output_columns = self._find_output_columns(output_shape)
        class_count = len(detect_stream.CLASS_NAMES)
        self.output_has_objectness = self.output_columns != class_count + 4
        input_dtype = np.dtype(self.input_details[0]["dtype"]).name
        output_dtype = np.dtype(self.output_details[0]["dtype"]).name
        self.description = (
            f"{self.name} / {self.model_path.name} | input "
            f"{self.input_width}x{self.input_height} {input_dtype} | "
            f"output {output_dtype}"
        )

    @staticmethod
    def _find_output_columns(output_shape):
        dimensions = [value for value in output_shape if value > 1]
        if not dimensions:
            raise ValueError(f"Output TFLite không hợp lệ: {output_shape}")
        class_count = len(detect_stream.CLASS_NAMES)
        expected = {class_count + 4, class_count + 5}
        for value in dimensions:
            if value in expected:
                return value
        return min(dimensions)

    @staticmethod
    def _quantization(detail):
        scale, zero_point = detail.get("quantization", (0.0, 0))
        return float(scale), int(zero_point)

    def _prepare_input(self, padded):
        detail = self.input_details[0]
        dtype = np.dtype(detail["dtype"])
        model_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = model_image.astype(np.float32) / 255.0
        if np.issubdtype(dtype, np.floating):
            return np.expand_dims(normalized.astype(dtype), 0)

        scale, zero_point = self._quantization(detail)
        if scale > 0:
            quantized = np.rint(normalized / scale + zero_point)
        else:
            quantized = model_image
        limits = np.iinfo(dtype)
        quantized = np.clip(quantized, limits.min, limits.max).astype(dtype)
        return np.expand_dims(quantized, 0)

    def _decode_output(self, raw_output):
        detail = self.output_details[0]
        output = np.asarray(raw_output)
        if np.issubdtype(output.dtype, np.integer):
            scale, zero_point = self._quantization(detail)
            if scale > 0:
                output = (output.astype(np.float32) - zero_point) * scale
            else:
                output = output.astype(np.float32)

        output = np.squeeze(output)
        if output.ndim != 2:
            raise ValueError(
                f"Output model phải có 2 chiều sau khi bỏ batch, nhận {output.shape}"
            )
        if (
            output.shape[0] == self.output_columns
            and output.shape[1] != self.output_columns
        ):
            output = output.T
        if output.shape[1] != self.output_columns:
            raise ValueError(
                f"Không nhận diện được bố cục output {output.shape}; "
                f"dự kiến {self.output_columns} giá trị cho mỗi detection"
            )
        return output

    def process(self, frame):
        confidence, iou = detect_stream.get_detection_thresholds()
        started = time.perf_counter()
        padded, ratio, padding = detect_stream.letterbox(
            frame, (self.input_height, self.input_width)
        )
        input_info = self.input_details[0]
        tensor = self._prepare_input(padded)

        with self.lock:
            self.interpreter.set_tensor(input_info["index"], tensor)
            self.interpreter.invoke()
            raw_output = self.interpreter.get_tensor(
                self.output_details[0]["index"]
            ).copy()
        output = self._decode_output(raw_output)

        annotated, counts = detect_stream.postprocess_and_draw_result(
            output,
            ratio,
            padding,
            frame.copy(),
            input_shape=(self.input_height, self.input_width),
            output_has_objectness=self.output_has_objectness,
        )
        counts.update({
            "inference_time": round((time.perf_counter() - started) * 1000, 1),
            "confidence_threshold": confidence,
            "nms_threshold": iou,
            "backend": self.name,
            "model_path": str(self.model_path),
        })
        return draw_summary(annotated, counts), counts


class YoloDetector:
    name = BACKEND_YOLO

    def __init__(self, model_path):
        if YOLO is None:
            raise RuntimeError(
                "Chưa cài ultralytics. Hãy chạy: pip install ultralytics"
            )
        self.model_path = Path(model_path).resolve()
        self.model = YOLO(str(self.model_path))
        self.lock = threading.Lock()
        self.names = self.model.names
        self.description = f"{self.name} / {self.model_path.name}"

    def _class_name(self, class_id):
        if isinstance(self.names, dict):
            return str(self.names.get(class_id, f"class{class_id}"))
        if class_id < len(self.names):
            return str(self.names[class_id])
        return f"class{class_id}"

    def process(self, frame):
        confidence, iou = detect_stream.get_detection_thresholds()
        started = time.perf_counter()
        with self.lock:
            result = self.model.predict(
                source=frame,
                conf=confidence,
                iou=iou,
                imgsz=detect_stream.INPUT_SIZE,
                verbose=False,
            )[0]

        annotated = frame.copy()
        disease_count = 0
        healthy_count = 0
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            for coordinates, score, class_id in zip(
                xyxy, confidences, class_ids
            ):
                x1, y1, x2, y2 = [int(value) for value in coordinates]
                if class_id == 0:
                    disease_count += 1
                    color = (0, 0, 255)
                else:
                    healthy_count += 1
                    color = (0, 255, 0)

                label = f"{self._class_name(class_id)} {float(score):.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                text_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
                )[0]
                label_top = max(0, y1 - 26)
                cv2.rectangle(
                    annotated,
                    (x1, label_top),
                    (x1 + text_size[0] + 10, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 5, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        plants_detected = 0 if boxes is None else len(boxes)
        stats = {
            "plants_detected": plants_detected,
            "disease_count": disease_count,
            "healthy_count": healthy_count,
            "inference_time": round((time.perf_counter() - started) * 1000, 1),
            "confidence_threshold": confidence,
            "nms_threshold": iou,
            "backend": self.name,
            "model_path": str(self.model_path),
        }
        return draw_summary(annotated, stats), stats


def create_detector(backend, model_path):
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {path}")
    if backend == BACKEND_TFLITE:
        if path.suffix.lower() != ".tflite":
            raise ValueError("TensorFlow Lite yêu cầu model có đuôi .tflite")
        return TensorFlowLiteDetector(path)
    if backend == BACKEND_YOLO:
        if path.suffix.lower() not in {".pt", ".onnx", ".engine"}:
            raise ValueError("YOLO hỗ trợ model .pt, .onnx hoặc .engine")
        return YoloDetector(path)
    raise ValueError(f"Backend không được hỗ trợ: {backend}")


class DetectionTestApp:
    POLL_INTERVAL_MS = 30

    def __init__(self, root):
        self.root = root
        self.root.title("Kiểm thử xử lý ảnh - RealSense / Ảnh 2D")
        self.root.geometry("1280x820")
        self.root.minsize(980, 680)

        self.stop_event = threading.Event()
        self.camera_stop_event = threading.Event()
        self.job_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue()

        self.state_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.generation = 0
        self.source_mode = "idle"
        self.camera_thread = None
        self.inference_thread = None
        self.model_loader_thread = None

        self.original_image = None
        self.display_image = None
        self.tk_image = None
        self.resize_after_id = None

        confidence, iou = load_detection_thresholds()
        detect_stream.set_detection_thresholds(confidence, iou)

        default_model_path = APP_DIR / detect_stream.MODEL_PATH
        self.active_detector = None
        self.initial_model_error = None
        try:
            self.active_detector = create_detector(
                BACKEND_TFLITE, default_model_path
            )
        except Exception as exc:
            self.initial_model_error = str(exc)

        self.backend_var = tk.StringVar(value=BACKEND_TFLITE)
        self.model_path_var = tk.StringVar(value=str(default_model_path))
        self.loaded_model_var = tk.StringVar(value="Chưa nạp model")
        self.confidence_var = tk.StringVar(value=f"{confidence:.2f}")
        self.iou_var = tk.StringVar(value=f"{iou:.2f}")
        self.status_var = tk.StringVar(value="Sẵn sàng")
        self.source_var = tk.StringVar(value="Chưa chọn")
        self.resolution_var = tk.StringVar(value="-")
        self.fps_var = tk.StringVar(value="0.0")
        self.inference_var = tk.StringVar(value="0 ms")
        self.plants_var = tk.StringVar(value="0")
        self.disease_var = tk.StringVar(value="0")
        self.healthy_var = tk.StringVar(value="0")
        self.active_thresholds_var = tk.StringVar(
            value=f"Confidence {confidence:.2f} | IoU {iou:.2f}"
        )
        if self.active_detector is not None:
            self.loaded_model_var.set(self.active_detector.description)

        self._build_ui()
        self._start_inference_worker()
        self.root.after(self.POLL_INTERVAL_MS, self._poll_worker_results)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        if self.active_detector is None:
            self._set_status(
                self.initial_model_error or "Chưa có model hoạt động",
                error=True,
            )
            self.camera_button.configure(state=tk.DISABLED)
            self.image_button.configure(state=tk.DISABLED)
        else:
            self._log(
                f"TensorFlow Lite sẵn sàng | input={detect_stream.INPUT_SIZE}px | "
                f"classes={', '.join(detect_stream.CLASS_NAMES)}"
            )

    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=0, sticky="nsew")
        controls.columnconfigure(0, weight=1)

        viewer = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        viewer.grid(row=0, column=1, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(1, weight=1)

        title = ttk.Label(
            controls,
            text="KIỂM THỬ NHẬN DIỆN",
            font=("Segoe UI", 16, "bold"),
        )
        title.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        model_box = ttk.LabelFrame(controls, text="Model nhận diện", padding=10)
        model_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        model_box.columnconfigure(0, weight=1)

        self.backend_combo = ttk.Combobox(
            model_box,
            textvariable=self.backend_var,
            values=(BACKEND_TFLITE, BACKEND_YOLO),
            state="readonly",
        )
        self.backend_combo.grid(row=0, column=0, columnspan=2, sticky="ew", pady=3)
        self.backend_combo.bind("<<ComboboxSelected>>", self._backend_changed)

        ttk.Entry(model_box, textvariable=self.model_path_var).grid(
            row=1, column=0, sticky="ew", padx=(0, 5), pady=3
        )
        ttk.Button(
            model_box,
            text="Chọn model",
            command=self.browse_model,
        ).grid(row=1, column=1, sticky="ew", pady=3)

        self.load_model_button = ttk.Button(
            model_box,
            text="Nạp / kiểm tra model",
            command=self.load_selected_model,
        )
        self.load_model_button.grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(5, 3)
        )
        ttk.Label(
            model_box,
            textvariable=self.loaded_model_var,
            foreground="#1565c0",
            wraplength=280,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))

        source_box = ttk.LabelFrame(controls, text="Nguồn ảnh", padding=10)
        source_box.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        source_box.columnconfigure(0, weight=1)

        self.camera_button = ttk.Button(
            source_box,
            text="▶ Bắt đầu RealSense",
            command=self.start_camera,
        )
        self.camera_button.grid(row=0, column=0, sticky="ew", pady=3)

        self.stop_camera_button = ttk.Button(
            source_box,
            text="■ Dừng camera",
            command=self.stop_camera,
            state=tk.DISABLED,
        )
        self.stop_camera_button.grid(row=1, column=0, sticky="ew", pady=3)

        self.image_button = ttk.Button(
            source_box,
            text="🖼 Chọn ảnh từ máy tính",
            command=self.open_image,
        )
        self.image_button.grid(row=2, column=0, sticky="ew", pady=3)

        self.save_button = ttk.Button(
            source_box,
            text="💾 Lưu ảnh kết quả",
            command=self.save_result,
            state=tk.DISABLED,
        )
        self.save_button.grid(row=3, column=0, sticky="ew", pady=3)

        threshold_box = ttk.LabelFrame(
            controls, text="Ngưỡng xử lý (0–1)", padding=10
        )
        threshold_box.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        threshold_box.columnconfigure(1, weight=1)

        ttk.Label(threshold_box, text="Confidence:").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=4
        )
        ttk.Spinbox(
            threshold_box,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.confidence_var,
            width=8,
        ).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(threshold_box, text="IoU / NMS:").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=4
        )
        ttk.Spinbox(
            threshold_box,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.iou_var,
            width=8,
        ).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Button(
            threshold_box,
            text="Áp dụng ngưỡng",
            command=self.apply_thresholds,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 2))

        ttk.Label(
            threshold_box,
            textvariable=self.active_thresholds_var,
            foreground="#1565c0",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        stats_box = ttk.LabelFrame(controls, text="Kết quả", padding=10)
        stats_box.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        stats_box.columnconfigure(1, weight=1)

        stats = (
            ("Model:", self.loaded_model_var),
            ("Nguồn:", self.source_var),
            ("Kích thước:", self.resolution_var),
            ("FPS camera:", self.fps_var),
            ("Inference:", self.inference_var),
            ("Số cây:", self.plants_var),
            ("Bị bệnh:", self.disease_var),
            ("Bình thường:", self.healthy_var),
        )
        for row, (label, variable) in enumerate(stats):
            ttk.Label(stats_box, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=2
            )
            ttk.Label(
                stats_box,
                textvariable=variable,
                font=("Segoe UI", 9, "bold"),
            ).grid(row=row, column=1, sticky="e", pady=2)

        log_box = ttk.LabelFrame(controls, text="Nhật ký", padding=6)
        log_box.grid(row=5, column=0, sticky="nsew")
        controls.rowconfigure(5, weight=1)
        self.log_text = tk.Text(
            log_box,
            width=38,
            height=10,
            state=tk.DISABLED,
            wrap="word",
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            viewer,
            textvariable=self.status_var,
            anchor="center",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.image_label = tk.Label(
            viewer,
            text="Chọn ảnh hoặc bắt đầu camera RealSense",
            bg="#111111",
            fg="white",
            font=("Segoe UI", 14),
            compound=tk.CENTER,
        )
        self.image_label.grid(row=1, column=0, sticky="nsew")
        self.image_label.bind("<Configure>", self._schedule_resize)

    def _start_inference_worker(self):
        self.inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="test-inference",
        )
        self.inference_thread.start()

    def _current_generation(self):
        with self.state_lock:
            return self.generation

    def _switch_source(self, source):
        with self.state_lock:
            self.generation += 1
            self.source_mode = source
            return self.generation

    def _is_current(self, generation, source=None):
        with self.state_lock:
            return (
                generation == self.generation
                and (source is None or source == self.source_mode)
            )

    def _put_latest_job(self, job):
        while True:
            try:
                self.job_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self.job_queue.put_nowait(job)
        except queue.Full:
            pass

    def _emit(self, event_type, **payload):
        self.result_queue.put({"type": event_type, **payload})

    def _backend_changed(self, _event=None):
        backend = self.backend_var.get()
        current_path = Path(self.model_path_var.get().strip())
        if backend == BACKEND_TFLITE:
            if current_path.suffix.lower() != ".tflite":
                self.model_path_var.set(str(APP_DIR / detect_stream.MODEL_PATH))
        elif current_path.suffix.lower() not in {".pt", ".onnx", ".engine"}:
            self.model_path_var.set("")

    def browse_model(self):
        backend = self.backend_var.get()
        if backend == BACKEND_TFLITE:
            filetypes = [("TensorFlow Lite", "*.tflite"), ("Tất cả tệp", "*.*")]
        else:
            filetypes = [
                ("YOLO model", "*.pt *.onnx *.engine"),
                ("Tất cả tệp", "*.*"),
            ]
        filename = filedialog.askopenfilename(
            title=f"Chọn model {backend}",
            initialdir=str(APP_DIR),
            filetypes=filetypes,
        )
        if filename:
            self.model_path_var.set(filename)

    def load_selected_model(self):
        if (
            self.model_loader_thread is not None
            and self.model_loader_thread.is_alive()
        ):
            return
        backend = self.backend_var.get()
        model_path = self.model_path_var.get().strip()
        if not model_path:
            messagebox.showwarning("Chưa chọn model", "Hãy chọn file model trước.")
            return

        self.load_model_button.configure(state=tk.DISABLED)
        self.backend_combo.configure(state=tk.DISABLED)
        self.loaded_model_var.set(f"Đang nạp {backend}...")
        self._set_status(f"Đang nạp và kiểm tra model {backend}...")
        self.model_loader_thread = threading.Thread(
            target=self._load_model_worker,
            args=(backend, model_path),
            daemon=True,
            name="test-model-loader",
        )
        self.model_loader_thread.start()

    def _load_model_worker(self, backend, model_path):
        try:
            detector = create_detector(backend, model_path)
            self._emit(
                "model_loaded",
                detector=detector,
                backend=backend,
                model_path=str(Path(model_path).resolve()),
            )
        except Exception as exc:
            self._emit("model_error", message=str(exc))

    def _get_active_detector(self):
        with self.model_lock:
            return self.active_detector

    def _queue_current_image_again(self):
        if self.original_image is None:
            return
        generation = self._switch_source("image")
        self._set_status("Đang xử lý lại ảnh với model mới...")
        self._put_latest_job({
            "generation": generation,
            "source": "image",
            "frame": self.original_image.copy(),
            "fps": 0.0,
            "filename": "Ảnh đã chọn",
        })

    def start_camera(self):
        if self._get_active_detector() is None:
            messagebox.showwarning("Chưa có model", "Hãy nạp model trước.")
            return
        if rs is None:
            messagebox.showerror(
                "Thiếu RealSense SDK",
                "Không tìm thấy pyrealsense2. Hãy cài SDK trước khi thử camera.",
            )
            return
        if self.camera_thread is not None and self.camera_thread.is_alive():
            return

        generation = self._switch_source("camera")
        self.camera_stop_event.clear()
        self.original_image = None
        self.camera_button.configure(state=tk.DISABLED)
        self.stop_camera_button.configure(state=tk.NORMAL)
        self.image_button.configure(state=tk.NORMAL)
        self._set_status("Đang khởi động RealSense...")

        self.camera_thread = threading.Thread(
            target=self._camera_loop,
            args=(generation,),
            daemon=True,
            name="test-realsense-capture",
        )
        self.camera_thread.start()

    def stop_camera(self):
        self.camera_stop_event.set()
        with self.state_lock:
            if self.source_mode == "camera":
                self.generation += 1
                self.source_mode = "idle"
        self.stop_camera_button.configure(state=tk.DISABLED)
        self.camera_button.configure(state=tk.DISABLED)
        self._set_status("Đang dừng camera...")

    def _camera_loop(self, generation):
        pipeline = None
        try:
            pipeline = rs.pipeline()
            camera_config = rs.config()
            camera_config.enable_stream(
                rs.stream.color,
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
                rs.format.bgr8,
                CAMERA_FPS,
            )
            pipeline.start(camera_config)
            self._emit("camera_started", generation=generation)

            frame_counter = 0
            fps = 0.0
            fps_started = time.monotonic()

            while (
                not self.stop_event.is_set()
                and not self.camera_stop_event.is_set()
                and self._is_current(generation, "camera")
            ):
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=500)
                except RuntimeError:
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data()).copy()
                frame_counter += 1
                elapsed = time.monotonic() - fps_started
                if elapsed >= 1.0:
                    fps = frame_counter / elapsed
                    frame_counter = 0
                    fps_started = time.monotonic()

                self._put_latest_job({
                    "generation": generation,
                    "source": "camera",
                    "frame": frame,
                    "fps": fps,
                })
        except Exception as exc:
            self._emit(
                "error",
                generation=generation,
                message=f"Không thể sử dụng RealSense: {exc}",
            )
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            self._emit("camera_stopped", generation=generation)

    def open_image(self):
        if self._get_active_detector() is None:
            messagebox.showwarning("Chưa có model", "Hãy nạp model trước.")
            return
        filename = filedialog.askopenfilename(
            title="Chọn ảnh để xử lý",
            filetypes=[
                ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Tất cả tệp", "*.*"),
            ],
        )
        if not filename:
            return

        try:
            raw = np.fromfile(filename, dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception as exc:
            messagebox.showerror("Không đọc được ảnh", str(exc))
            return
        if frame is None:
            messagebox.showerror(
                "Không đọc được ảnh",
                "Tệp đã chọn không phải định dạng ảnh được hỗ trợ.",
            )
            return

        self.camera_stop_event.set()
        generation = self._switch_source("image")
        self.original_image = frame.copy()
        self.camera_button.configure(state=tk.DISABLED)
        self.stop_camera_button.configure(state=tk.DISABLED)
        self._set_status(f"Đang xử lý {Path(filename).name}...")
        self._put_latest_job({
            "generation": generation,
            "source": "image",
            "frame": frame,
            "fps": 0.0,
            "filename": Path(filename).name,
        })

    def apply_thresholds(self):
        try:
            confidence = float(self.confidence_var.get())
            iou = float(self.iou_var.get())
            detect_stream.set_detection_thresholds(confidence, iou)
        except ValueError as exc:
            messagebox.showerror("Ngưỡng không hợp lệ", str(exc))
            return

        self.confidence_var.set(f"{confidence:.2f}")
        self.iou_var.set(f"{iou:.2f}")
        self.active_thresholds_var.set(
            f"Confidence {confidence:.2f} | IoU {iou:.2f}"
        )
        self._log(
            f"Đã áp dụng Confidence={confidence:.2f}, IoU={iou:.2f}"
        )

        if self.original_image is not None:
            generation = self._switch_source("image")
            self._set_status("Đang xử lý lại ảnh với ngưỡng mới...")
            self._put_latest_job({
                "generation": generation,
                "source": "image",
                "frame": self.original_image.copy(),
                "fps": 0.0,
                "filename": "Ảnh đã chọn",
            })

    def _inference_loop(self):
        while not self.stop_event.is_set():
            try:
                job = self.job_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                break
            if not self._is_current(job["generation"], job["source"]):
                continue

            try:
                detector = self._get_active_detector()
                if detector is None:
                    raise RuntimeError("Chưa có model hoạt động")
                processed, stats = detector.process(job["frame"])
                if job["source"] == "camera":
                    cv2.putText(
                        processed,
                        f"FPS: {job['fps']:.1f}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                if self._is_current(job["generation"], job["source"]):
                    self._emit(
                        "frame",
                        generation=job["generation"],
                        source=job["source"],
                        frame=processed,
                        stats=stats,
                        fps=job["fps"],
                        filename=job.get("filename", ""),
                    )
            except Exception as exc:
                self._emit(
                    "error",
                    generation=job["generation"],
                    message=f"Lỗi xử lý ảnh: {exc}",
                )

    def _poll_worker_results(self):
        try:
            while True:
                event = self.result_queue.get_nowait()
                event_type = event["type"]
                generation = event.get("generation")

                if event_type == "model_loaded":
                    with self.model_lock:
                        self.active_detector = event["detector"]
                    self.loaded_model_var.set(event["detector"].description)
                    self.load_model_button.configure(state=tk.NORMAL)
                    self.backend_combo.configure(state="readonly")
                    self.camera_button.configure(state=tk.NORMAL)
                    self.image_button.configure(state=tk.NORMAL)
                    self._set_status("Model đã nạp và sẵn sàng kiểm tra")
                    self._log(
                        f"Đã nạp {event['backend']}: {event['model_path']}"
                    )
                    self._queue_current_image_again()
                elif event_type == "model_error":
                    active_detector = self._get_active_detector()
                    if active_detector is None:
                        self.loaded_model_var.set("Chưa có model hoạt động")
                        self.camera_button.configure(state=tk.DISABLED)
                        self.image_button.configure(state=tk.DISABLED)
                    else:
                        self.loaded_model_var.set(active_detector.description)
                    self.load_model_button.configure(state=tk.NORMAL)
                    self.backend_combo.configure(state="readonly")
                    self._set_status(
                        f"Không thể nạp model: {event['message']}", error=True
                    )
                    self._log(f"Nạp model thất bại: {event['message']}")
                elif event_type == "camera_started":
                    if self._is_current(generation, "camera"):
                        self._set_status("RealSense đang hoạt động")
                        self._log("Đã kết nối RealSense 640x480 @ 30 FPS")
                elif event_type == "camera_stopped":
                    if self.source_mode != "camera":
                        self.camera_button.configure(state=tk.NORMAL)
                        self.stop_camera_button.configure(state=tk.DISABLED)
                    self._log("Camera đã dừng")
                elif event_type == "error":
                    if generation is None or generation == self._current_generation():
                        self._set_status(event["message"], error=True)
                        self._log(event["message"])
                        self.camera_button.configure(state=tk.NORMAL)
                        self.stop_camera_button.configure(state=tk.DISABLED)
                elif event_type == "frame":
                    if self._is_current(generation, event["source"]):
                        self._show_result(event)
        except queue.Empty:
            pass

        if not self.stop_event.is_set():
            self.root.after(self.POLL_INTERVAL_MS, self._poll_worker_results)

    def _show_result(self, event):
        frame = event["frame"]
        stats = event["stats"]
        height, width = frame.shape[:2]
        self.display_image = frame.copy()

        source_text = (
            "RealSense"
            if event["source"] == "camera"
            else event.get("filename") or "Ảnh 2D"
        )
        self.source_var.set(source_text)
        self.resolution_var.set(f"{width} × {height}")
        self.fps_var.set(f"{event['fps']:.1f}")
        self.inference_var.set(f"{stats['inference_time']:.1f} ms")
        self.plants_var.set(str(stats["plants_detected"]))
        self.disease_var.set(str(stats["disease_count"]))
        self.healthy_var.set(str(stats["healthy_count"]))
        self.active_thresholds_var.set(
            f"Confidence {stats['confidence_threshold']:.2f} | "
            f"IoU {stats['nms_threshold']:.2f}"
        )
        self.save_button.configure(state=tk.NORMAL)
        self._render_image()

        if event["source"] == "camera":
            self._set_status("Đang xử lý camera RealSense theo thời gian thực")
        else:
            self._set_status("Xử lý ảnh hoàn tất")
            self.camera_button.configure(state=tk.NORMAL)
            self._log(
                f"Ảnh hoàn tất: {stats['plants_detected']} cây, "
                f"{stats['inference_time']:.1f} ms"
            )

    def _schedule_resize(self, _event=None):
        if self.resize_after_id is not None:
            self.root.after_cancel(self.resize_after_id)
        self.resize_after_id = self.root.after(100, self._render_image)

    def _render_image(self):
        self.resize_after_id = None
        if self.display_image is None:
            return

        rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_width = max(320, self.image_label.winfo_width() - 12)
        max_height = max(240, self.image_label.winfo_height() - 12)
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.tk_image, text="")

    def save_result(self):
        if self.display_image is None:
            return
        filename = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
        )
        if not filename:
            return

        extension = Path(filename).suffix.lower()
        encode_extension = ".png" if extension == ".png" else ".jpg"
        params = [] if encode_extension == ".png" else [cv2.IMWRITE_JPEG_QUALITY, 92]
        ok, encoded = cv2.imencode(encode_extension, self.display_image, params)
        if not ok:
            messagebox.showerror("Lỗi", "Không thể mã hóa ảnh kết quả")
            return
        try:
            encoded.tofile(filename)
            self._log(f"Đã lưu kết quả: {filename}")
        except OSError as exc:
            messagebox.showerror("Không thể lưu ảnh", str(exc))

    def _set_status(self, message, error=False):
        self.status_var.set(("LỖI: " if error else "") + message)

    def _log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 200:
            self.log_text.delete("1.0", "30.0")
        self.log_text.configure(state=tk.DISABLED)

    def close(self):
        self.stop_event.set()
        self.camera_stop_event.set()
        try:
            self.job_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        if self.inference_thread is not None and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        self.root.destroy()


def main():
    root = tk.Tk()
    DetectionTestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
