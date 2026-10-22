from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)
from functools import wraps
import base64
import hmac
import threading
import time
import json
import math
import os
import secrets
import cv2
import numpy as np
from Read_Serial import (
    SerialData,
    execute_route_commands,
    start_serial_thread,
    validate_route_commands,
)
from config import (AUTO_DRIVE_ENABLED, FLASK_HOST, FLASK_PORT, GOOGLE_MAPS_API_KEY, LIDAR_PORT, LIDAR_BAUDRATE,
                    LIDAR_OBSTACLE_DISTANCE, LIDAR_DETECTION_ANGLE_MIN, LIDAR_DETECTION_ANGLE_MAX)
from control_link import ControlLink
from mdns_service import MdnsAdvertiser
from serial_port_probe import (
    DEVICE_LABELS,
    available_serial_ports,
    port_matches_device_usb,
    probe_serial_port,
)
import detect_stream
import Read_lidar
import config
import arm_control
import depth_stream

# Tạo Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("ROBOT_FLASK_SECRET_KEY", secrets.token_hex(32))
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

CONFIG_PASSWORD = "0123"
MAX_IMAGE_UPLOAD_BYTES = 10 * 1024 * 1024

# =====================================================
# Runtime Settings - Có thể thay đổi mà không cần restart
# =====================================================
runtime_settings = {
    "GPS_IMU_PORT": config.SERIAL_PORT,
    "CONTROL_PORT": config.SERIAL_PORT_CONTROL,
    "LIDAR_PORT": config.LIDAR_PORT,
    "TIME_PER_METER_SEC": config.TIME_PER_METER_SEC,
    "DEFAULT_LATITUDE": config.DEFAULT_LATITUDE,
    "DEFAULT_LONGITUDE": config.DEFAULT_LONGITUDE,
    "LIDAR_MAX_POINTS": config.LIDAR_MAX_POINTS,
    "LIDAR_DOT_LIFETIME": config.LIDAR_DOT_LIFETIME,
    "LIDAR_OBSTACLE_DISTANCE": config.LIDAR_OBSTACLE_DISTANCE,
    "LIDAR_DETECTION_ANGLE_MIN": config.LIDAR_DETECTION_ANGLE_MIN,
    "LIDAR_DETECTION_ANGLE_MAX": config.LIDAR_DETECTION_ANGLE_MAX,
    "IMAGE_2D_ENABLED": config.IMAGE_2D_ENABLED,
    "DETECTION_CONFIDENCE_THRESHOLD": config.DETECTION_CONFIDENCE_THRESHOLD,
    "DETECTION_IOU_THRESHOLD": config.DETECTION_IOU_THRESHOLD,
}
settings_lock = threading.Lock()

# Load settings từ file JSON nếu tồn tại
def load_settings_from_json():
    """Load settings từ settings.json"""
    global runtime_settings
    try:
        if os.path.exists("settings.json"):
            with open("settings.json", "r", encoding="utf-8") as f:
                loaded = json.load(f)
                with settings_lock:
                    runtime_settings.update(loaded)
                print(f"[Settings] Loaded from settings.json: {runtime_settings}")
        else:
            print("[Settings] No settings.json found, using config.py defaults")
    except Exception as e:
        print(f"[Settings] Error loading settings.json: {e}")

def save_settings_to_json():
    """Lưu settings vào settings.json"""
    try:
        with settings_lock:
            with open("settings.json", "w", encoding="utf-8") as f:
                json.dump(runtime_settings, f, indent=4)
        print(f"[Settings] Saved to settings.json: {runtime_settings}")
        return True
    except Exception as e:
        print(f"[Settings] Error saving settings.json: {e}")
        return False

def get_setting(key):
    """Lấy giá trị setting (thread-safe)"""
    with settings_lock:
        return runtime_settings.get(key)

def update_setting(key, value):
    """Cập nhật setting (thread-safe)"""
    with settings_lock:
        runtime_settings[key] = value


def parse_boolean(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off", ""}:
        return False
    raise ValueError("Giá trị bật/tắt xử lý ảnh 2D không hợp lệ")


def parse_config_values(data):
    """Parse and validate values received from the protected config page."""
    values = {
        "GPS_IMU_PORT": str(data.get("GPS_IMU_PORT", config.SERIAL_PORT)).strip(),
        "CONTROL_PORT": str(data.get("CONTROL_PORT", config.SERIAL_PORT_CONTROL)).strip(),
        "LIDAR_PORT": str(data.get("LIDAR_PORT", config.LIDAR_PORT)).strip(),
        "TIME_PER_METER_SEC": float(data.get("TIME_PER_METER_SEC", 1.0)),
        "DEFAULT_LATITUDE": float(data.get("DEFAULT_LATITUDE", config.DEFAULT_LATITUDE)),
        "DEFAULT_LONGITUDE": float(data.get("DEFAULT_LONGITUDE", config.DEFAULT_LONGITUDE)),
        "LIDAR_MAX_POINTS": int(data.get("LIDAR_MAX_POINTS", 1500)),
        "LIDAR_DOT_LIFETIME": float(data.get("LIDAR_DOT_LIFETIME", 1.0)),
        "LIDAR_OBSTACLE_DISTANCE": int(data.get("LIDAR_OBSTACLE_DISTANCE", 400)),
        "LIDAR_DETECTION_ANGLE_MIN": int(data.get("LIDAR_DETECTION_ANGLE_MIN", -20)),
        "LIDAR_DETECTION_ANGLE_MAX": int(data.get("LIDAR_DETECTION_ANGLE_MAX", 20)),
        "IMAGE_2D_ENABLED": parse_boolean(
            data.get("IMAGE_2D_ENABLED"),
            default=bool(get_setting("IMAGE_2D_ENABLED")),
        ),
        "DETECTION_CONFIDENCE_THRESHOLD": float(data.get(
            "DETECTION_CONFIDENCE_THRESHOLD",
            get_setting("DETECTION_CONFIDENCE_THRESHOLD"),
        )),
        "DETECTION_IOU_THRESHOLD": float(data.get(
            "DETECTION_IOU_THRESHOLD",
            get_setting("DETECTION_IOU_THRESHOLD"),
        )),
    }

    if not -90.0 <= values["DEFAULT_LATITUDE"] <= 90.0:
        raise ValueError("Vĩ độ mặc định phải nằm trong khoảng -90 đến 90")
    if not -180.0 <= values["DEFAULT_LONGITUDE"] <= 180.0:
        raise ValueError("Kinh độ mặc định phải nằm trong khoảng -180 đến 180")
    confidence = values["DETECTION_CONFIDENCE_THRESHOLD"]
    iou = values["DETECTION_IOU_THRESHOLD"]
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("Ngưỡng phát hiện phải nằm trong khoảng 0 đến 1")
    if not math.isfinite(iou) or not 0.0 <= iou <= 1.0:
        raise ValueError("Ngưỡng IoU phải nằm trong khoảng 0 đến 1")
    selected_ports = [
        values["GPS_IMU_PORT"],
        values["CONTROL_PORT"],
        values["LIDAR_PORT"],
    ]
    if any(not port for port in selected_ports):
        raise ValueError("Phải chọn đầy đủ cổng COM cho ba thiết bị")
    if len(set(port.upper() for port in selected_ports)) != len(selected_ports):
        raise ValueError("Mỗi thiết bị phải sử dụng một cổng COM riêng")
    return values


def apply_config_values(values):
    with settings_lock:
        runtime_settings.update(values)
    detect_stream.set_detection_thresholds(
        values.get(
            "DETECTION_CONFIDENCE_THRESHOLD",
            get_setting("DETECTION_CONFIDENCE_THRESHOLD"),
        ),
        values.get(
            "DETECTION_IOU_THRESHOLD",
            get_setting("DETECTION_IOU_THRESHOLD"),
        ),
    )


def config_login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("config_authenticated"):
            if request.path == "/config":
                return redirect(url_for("config_login"))
            return jsonify({
                "success": False,
                "message": "Vui lòng đăng nhập để truy cập cấu hình",
            }), 401
        return view(*args, **kwargs)
    return wrapped

# Load settings khi khởi động
load_settings_from_json()
detect_stream.set_detection_thresholds(
    get_setting("DETECTION_CONFIDENCE_THRESHOLD"),
    get_setting("DETECTION_IOU_THRESHOLD"),
)

ACTIVE_SERIAL_PORTS = {
    "gps_imu": get_setting("GPS_IMU_PORT"),
    "controller": get_setting("CONTROL_PORT"),
    "lidar": get_setting("LIDAR_PORT"),
}

SERIAL_SETTING_KEYS = {
    "gps_imu": "GPS_IMU_PORT",
    "controller": "CONTROL_PORT",
    "lidar": "LIDAR_PORT",
}


def serial_ports_restart_required(values):
    return any(
        values[setting_key].upper() != ACTIVE_SERIAL_PORTS[device_type].upper()
        for device_type, setting_key in SERIAL_SETTING_KEYS.items()
    )


def require_verified_serial_changes(values):
    verified_ports = session.get("verified_serial_ports", {})
    for device_type, setting_key in SERIAL_SETTING_KEYS.items():
        selected_port = values[setting_key]
        if selected_port.upper() == ACTIVE_SERIAL_PORTS[device_type].upper():
            continue
        verified_port = verified_ports.get(device_type, "")
        if verified_port.upper() != selected_port.upper():
            raise ValueError(
                f"Hãy kiểm tra {DEVICE_LABELS[device_type]} trên {selected_port} "
                "trước khi lưu"
            )

# Tạo đối tượng lưu dữ liệu GPS
shared_data = SerialData()

# Chỉ ControlLink được quyền mở và ghi cổng điều khiển ESP32 chính.
control_link = ControlLink(shared_data, port=ACTIVE_SERIAL_PORTS["controller"])
control_link.start()

# Bắt đầu luồng đọc Serial nền
t = threading.Thread(
    target=start_serial_thread,
    args=(shared_data, ACTIVE_SERIAL_PORTS["gps_imu"]),
    daemon=True,
)
t.start()

# Background thread để tự động kiểm tra LIDAR và cập nhật obstacle status
def lidar_monitor_thread(shared_data):
    """Tính obstacle ở server; mất/stale Lidar luôn là trạng thái dừng."""
    print("[LIDAR Monitor] Background thread started")
    last_config_print = 0
    last_state = None
    
    while True:
        try:
            points = Read_lidar.lidar_data.get_current_points()
            angle_min = get_setting("LIDAR_DETECTION_ANGLE_MIN")
            angle_max = get_setting("LIDAR_DETECTION_ANGLE_MAX")
            obstacle_dist = get_setting("LIDAR_OBSTACLE_DISTANCE")
            lidar_valid = Read_lidar.is_data_fresh()

            current_time = time.monotonic()
            # The values rarely change; avoid filling journald during long runs.
            if current_time - last_config_print > 300:
                print(f"[LIDAR Monitor] Current config: Distance={obstacle_dist}mm, Angle=[{angle_min}° to {angle_max}°]")
                last_config_print = current_time

            min_distance = float('inf')
            has_obstacle = False

            if lidar_valid:
                for point in points:
                    angle = point["angle"] % 360.0
                    distance = point["distance"]

                    if angle_min < 0 <= angle_max:
                        is_in_zone = angle <= angle_max or angle >= 360 + angle_min
                    else:
                        is_in_zone = angle_min <= angle <= angle_max

                    if is_in_zone:
                        min_distance = min(min_distance, distance)
                        if distance < obstacle_dist:
                            has_obstacle = True

                distance_value = min_distance if min_distance != float('inf') else 9999
                shared_data.set_lidar_obstacle(
                    has_obstacle,
                    distance_value,
                    valid=True,
                )
                state = "obstacle" if has_obstacle else "clear"
            else:
                # Sensor state is unknown, therefore MANUAL_GUARD must stop.
                shared_data.set_lidar_obstacle(True, 9999, valid=False)
                state = "sensor_invalid"

            if state != last_state:
                if state == "obstacle":
                    print(f"[LIDAR Monitor] ⚠️ OBSTACLE: {int(distance_value)}mm")
                elif state == "clear":
                    print("[LIDAR Monitor] ✅ Clear; manual-allow restored")
                else:
                    print("[LIDAR Monitor] ⚠️ Sensor missing/stale; movement denied")
                last_state = state

            time.sleep(0.1)
            
        except Exception as e:
            print(f"[LIDAR Monitor] Error: {e}")
            # Never preserve a previous allow decision after the monitor fails.
            shared_data.set_lidar_obstacle(True, 9999, valid=False)
            last_state = "monitor_error"
            time.sleep(0.5)

# Khởi động LIDAR monitor thread
lidar_monitor = threading.Thread(target=lidar_monitor_thread, args=(shared_data,), daemon=True)
lidar_monitor.start()

# Đọc file HTML từ map.html
with open("map.html", "r", encoding="utf-8") as f:
    html_template = f.read()

# Đọc file HTML từ arm_web.html
with open("arm_web.html", "r", encoding="utf-8") as f:
    arm_html_template = f.read()

# Đọc file HTML từ depth_web.html
with open("depth_web.html", "r", encoding="utf-8") as f:
    depth_html_template = f.read()

# Đọc file HTML từ detect_web.html
with open("detect_web.html", "r", encoding="utf-8") as f:
    detect_html_template = f.read()

# Đọc file HTML từ lidar_web.html
with open("lidar_web.html", "r", encoding="utf-8") as f:
    lidar_html_template = f.read()

# Đọc file HTML từ config_web.html
with open("config_web.html", "r", encoding="utf-8") as f:
    config_html_template = f.read()

with open("config_login.html", "r", encoding="utf-8") as f:
    config_login_html_template = f.read()

@app.route("/")
def index():
    # render_template_string để chèn API key động vào file ngoài
    return render_template_string(
        html_template,
        GOOGLE_MAPS_API_KEY=GOOGLE_MAPS_API_KEY,
        DEFAULT_LATITUDE=get_setting("DEFAULT_LATITUDE"),
        DEFAULT_LONGITUDE=get_setting("DEFAULT_LONGITUDE"),
    )


@app.route("/config/login", methods=["GET", "POST"])
def config_login():
    if session.get("config_authenticated"):
        return redirect(url_for("config_page"))

    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        if hmac.compare_digest(password, CONFIG_PASSWORD):
            session["config_authenticated"] = True
            return redirect(url_for("config_page"))
        error = "Mật khẩu không đúng"

    return render_template_string(config_login_html_template, error=error)

@app.route("/config")
@config_login_required
def config_page():
    # Trang cấu hình
    return render_template_string(config_html_template)


@app.route("/config/logout", methods=["POST"])
@config_login_required
def config_logout():
    session.pop("config_authenticated", None)
    return redirect(url_for("index"))

@app.route("/arm")
def arm_page():
    """Trang điều khiển cánh tay robot (4 servo J1-J4)"""
    return render_template_string(arm_html_template)

@app.route("/setArmAngles", methods=["POST"])
def set_arm_angles():
    """Nhận góc 4 servo J1-J4 từ web, lưu vào shared_data và điều khiển GPIO trực tiếp"""
    try:
        j1 = int(float(request.form.get("j1", 90)))
        j2 = int(float(request.form.get("j2", 90)))
        j3 = int(float(request.form.get("j3", 90)))
        j4 = int(float(request.form.get("j4", 90)))

        # Giới hạn trong [0, 180]
        j1 = max(0, min(180, j1))
        j2 = max(0, min(180, j2))
        j3 = max(0, min(180, j3))
        j4 = max(0, min(180, j4))

        shared_data.set_arm_angles([j1, j2, j3, j4])

        # Điều khiển servo trực tiếp qua GPIO Raspberry Pi
        arm_control.set_all_angles([j1, j2, j3, j4])

        mode = arm_control.get_mode()
        if mode == "simulation":
            return Response(
                f"[Simulation] J1={j1}° J2={j2}° J3={j3}° J4={j4}°",
                mimetype="text/plain"
            )
        return Response(
            f"Góc đã gửi ({mode}): J1={j1}° J2={j2}° J3={j3}° J4={j4}°",
            mimetype="text/plain"
        )
    except Exception as e:
        print(f"[ARM] Error: {e}")
        return Response(f"Lỗi: {e}", status=500, mimetype="text/plain")

@app.route("/getArmAngles")
def get_arm_angles():
    """Trả về góc hiện tại của 4 servo dạng JSON array [j1, j2, j3, j4]"""
    angles = shared_data.get_arm_angles()
    return jsonify(angles)

@app.route("/detection")
def detection_page():
    # Trang detection
    return render_template_string(detect_html_template)

@app.route("/lidar")
def lidar_page():
    # Trang LIDAR - Tự động bắt đầu LIDAR
    # Nếu chưa chạy thì khởi động
    if not Read_lidar.get_status()["connected"]:
        Read_lidar.start_lidar_thread(ACTIVE_SERIAL_PORTS["lidar"], LIDAR_BAUDRATE)
    return render_template_string(lidar_html_template)

@app.route("/getGpsData")
def get_gps_data():
    snap = shared_data.snapshot()
    if snap.get("gps_valid"):
        latitude = snap.get("lat")
        longitude = snap.get("lon")
    else:
        latitude = get_setting("DEFAULT_LATITUDE")
        longitude = get_setting("DEFAULT_LONGITUDE")
    return f"{latitude},{longitude}"

@app.route("/postData", methods=["POST"])
def post_data():
    dis = request.form.get("dis", "")
    dir_ = request.form.get("dir", "")
    dir_val = request.form.get("dir_value", "")
    print(f"[POST] dis={dis}")
    print(f"[POST] dir={dir_}")
    print(f"[POST] dir_value={dir_val}")
    return Response("Data received", mimetype="text/plain")

@app.route("/getYaw")
def get_yaw():
    yaw = shared_data.snapshot().get("yaw", 0.0)
    return f"{yaw:.2f}"

# Route bắt đầu thực thi lệnh điều khiển
@app.route("/startRoute", methods=["POST"])
def start_route():
    if not AUTO_DRIVE_ENABLED:
        shared_data.set_route_state(stopped=True, running=False, paused=False)
        shared_data.set_move_cmd(0)
        control_link.request_stop()
        return Response(
            "AUTO_TODO: chuyển động và quay tự động chưa được mở khóa",
            status=503,
            mimetype="text/plain",
        )

    if shared_data.get_route_state()["running"]:
        return Response("Một hành trình AUTO đang chạy", status=409, mimetype="text/plain")

    try:
        dis = request.form.get("dis", "")
        dir_ = request.form.get("dir", "")
        dir_val = request.form.get("dir_value", "")
        dis_list, dir_list, dir_value_list = validate_route_commands(
            [float(x) for x in dis.split(",") if x],
            [int(x) for x in dir_.split(",") if x],
            [float(x) for x in dir_val.split(",") if x],
        )
    except ValueError as exc:
        return Response(f"Dữ liệu hành trình không hợp lệ: {exc}", status=400, mimetype="text/plain")

    controller_status = control_link.get_status()
    if not controller_status["connected"] or not controller_status["identified"]:
        return Response(
            "ESP32 điều khiển chưa kết nối hoặc chưa được nhận diện",
            status=503,
            mimetype="text/plain",
        )
    if not shared_data.imu_is_fresh():
        return Response("Dữ liệu IMU chưa sẵn sàng", status=503, mimetype="text/plain")
    lidar_status = shared_data.get_lidar_obstacle()
    if not lidar_status["valid"]:
        return Response("Dữ liệu LIDAR chưa sẵn sàng", status=503, mimetype="text/plain")
    if lidar_status["detected"]:
        return Response("Đang có vật cản phía trước", status=409, mimetype="text/plain")

    if not shared_data.try_begin_route(len(dis_list)):
        return Response("Một hành trình AUTO đang chạy", status=409, mimetype="text/plain")

    print(f"[START ROUTE] dis={dis_list}")
    print(f"[START ROUTE] dir={dir_list}")
    print(f"[START ROUTE] dir_value={dir_value_list}")
    try:
        threading.Thread(
            target=execute_route_commands,
            args=(shared_data, dis_list, dir_list, dir_value_list, True),
            daemon=True,
            name="auto-route",
        ).start()
    except Exception:
        shared_data.set_move_cmd(0)
        shared_data.set_route_state(running=False, paused=False, stopped=True)
        control_link.request_stop()
        raise
    return Response("Route started", mimetype="text/plain")

@app.route("/pauseRoute", methods=["POST"])
def pause_route():
    """Tạm dừng route"""
    shared_data.set_route_state(paused=True)
    shared_data.set_move_cmd(0)
    print("[PAUSE ROUTE] Route paused")
    return Response("Route paused", mimetype="text/plain")

@app.route("/resumeRoute", methods=["POST"])
def resume_route():
    """Tiếp tục route sau khi tạm dừng"""
    shared_data.set_route_state(paused=False)
    print("[RESUME ROUTE] Route resumed")
    return Response("Route resumed", mimetype="text/plain")

@app.route("/stopRoute", methods=["POST"])
def stop_route():
    """Dừng hẳn route"""
    shared_data.set_route_state(stopped=True, running=False, paused=False)
    shared_data.set_move_cmd(0)
    # Không ghi serial tại request thread. Sender duy nhất sẽ phát STOP ở
    # heartbeat kế tiếp và giữ STOP đủ lâu qua request_stop().
    control_link.request_stop()
    print("[STOP ROUTE] Route stopped")
    return Response("Route stopped", mimetype="text/plain")

@app.route("/getRouteStatus")
def get_route_status():
    """Lấy trạng thái route hiện tại"""
    status = shared_data.get_route_state()
    return jsonify(status)

# ===================== DETECTION ROUTES =====================
@app.route("/video_feed")
def video_feed():
    """Video streaming route"""
    if not detect_stream.init_camera():
        return Response("Camera not available", status=503)
    return Response(detect_stream.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start_detection", methods=["POST"])
def start_detection():
    """Enable object detection"""
    detect_stream.set_detection_enabled(True)
    return Response("Detection started", mimetype="text/plain")

@app.route("/stop_detection", methods=["POST"])
def stop_detection():
    """Disable object detection"""
    detect_stream.set_detection_enabled(False)
    return Response("Detection stopped", mimetype="text/plain")

@app.route("/detection_stats")
def detection_stats():
    """Get detection statistics"""
    stats = detect_stream.get_stats()
    return jsonify(stats)


@app.route("/detection_options")
def detection_options():
    """Public feature flags required to render the detection page."""
    confidence_threshold, iou_threshold = detect_stream.get_detection_thresholds()
    return jsonify({
        "image_2d_enabled": bool(get_setting("IMAGE_2D_ENABLED")),
        "max_upload_mb": MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024),
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
    })


@app.route("/process_2d_image", methods=["POST"])
def process_2d_image():
    """Process an uploaded image in memory with the realtime model pipeline."""
    if not get_setting("IMAGE_2D_ENABLED"):
        return jsonify({
            "success": False,
            "message": "Chức năng xử lý ảnh 2D đang tắt trong cấu hình",
        }), 403

    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({
            "success": False,
            "message": "Vui lòng chọn một tệp ảnh",
        }), 400

    raw_image = uploaded.read(MAX_IMAGE_UPLOAD_BYTES + 1)
    if len(raw_image) > MAX_IMAGE_UPLOAD_BYTES:
        return jsonify({
            "success": False,
            "message": "Ảnh vượt quá giới hạn 10 MB",
        }), 413
    if not raw_image:
        return jsonify({
            "success": False,
            "message": "Tệp ảnh rỗng",
        }), 400

    image_array = np.frombuffer(raw_image, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({
            "success": False,
            "message": "Không thể đọc tệp; hãy chọn ảnh JPG, PNG, BMP hoặc WEBP",
        }), 400

    try:
        processed, stats = detect_stream.process_still_image(image)
        encoded_ok, encoded = cv2.imencode(
            ".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not encoded_ok:
            raise RuntimeError("Không thể mã hóa ảnh kết quả")
        return jsonify({
            "success": True,
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "mime_type": "image/jpeg",
            "stats": stats,
        })
    except RuntimeError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception as exc:
        print(f"[Detection] 2D image processing error: {exc}")
        return jsonify({
            "success": False,
            "message": "Không thể xử lý ảnh đã chọn",
        }), 500

@app.route("/camera_start", methods=["POST"])
def camera_start():
    """Start camera"""
    if detect_stream.init_camera():
        return Response("Camera started", mimetype="text/plain")
    return Response("Camera failed to start", status=500)

@app.route("/camera_stop", methods=["POST"])
def camera_stop():
    """Stop camera"""
    detect_stream.release_camera()
    detect_stream.set_detection_enabled(False)
    return Response("Camera stopped", mimetype="text/plain")

# ===================== DEPTH CAMERA ROUTES =====================
@app.route("/depth")
def depth_page():
    """Trang xem camera chiều sâu RealSense D435"""
    return render_template_string(depth_html_template)

@app.route("/depth_feed")
def depth_feed():
    """MJPEG stream từ depth camera – tự start pipeline nếu chưa chạy"""
    return Response(
        depth_stream.generate_depth_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route("/depth_start", methods=["POST"])
def depth_start():
    """Khởi động depth pipeline"""
    ok = depth_stream.start_depth()
    if ok:
        return Response("Depth camera started", mimetype="text/plain")
    return Response("RealSense not available", status=503, mimetype="text/plain")

@app.route("/depth_stop", methods=["POST"])
def depth_stop():
    """Dừng depth pipeline – giải phóng tài nguyên"""
    depth_stream.stop_depth()
    return Response("Depth camera stopped", mimetype="text/plain")

@app.route("/depth_status")
def depth_status():
    """Trả về trạng thái depth pipeline (dùng để poll FPS, khoảng cách từ web)"""
    return jsonify({
        "running": depth_stream.is_running(),
        "fps": depth_stream._last_fps if hasattr(depth_stream, '_last_fps') else 0,
        "center_dist": depth_stream._last_center_dist if hasattr(depth_stream, '_last_center_dist') else 0,
    })

# ===================== LIDAR ROUTES =====================
@app.route("/startLidar", methods=["POST"])
def start_lidar():
    """Bắt đầu đọc LIDAR"""
    # Sử dụng config từ file config.py
    active_lidar_port = ACTIVE_SERIAL_PORTS["lidar"]
    success = Read_lidar.start_lidar_thread(active_lidar_port, LIDAR_BAUDRATE)
    if success:
        return Response(f"LIDAR started on {active_lidar_port} @ {LIDAR_BAUDRATE}", mimetype="text/plain")
    else:
        return Response("LIDAR already running", mimetype="text/plain")

@app.route("/stopLidar", methods=["POST"])
def stop_lidar():
    """Dừng đọc LIDAR"""
    Read_lidar.stop_lidar()
    return Response("LIDAR stopped", mimetype="text/plain")

@app.route("/getLidarData")
def get_lidar_data():
    """Lấy dữ liệu LIDAR hiện tại"""
    points = Read_lidar.lidar_data.get_current_points()
    status = Read_lidar.get_status()
    return jsonify({
        'points': points,
        'connected': status['connected'],
        'data_fresh': status['data_fresh'],
        'last_point_age': status['last_point_age'],
    })

@app.route("/clearLidarData", methods=["POST"])
def clear_lidar_data():
    """Xóa tất cả dữ liệu LIDAR"""
    Read_lidar.lidar_data.clear_all()
    return Response("LIDAR data cleared", mimetype="text/plain")

@app.route("/getLidarObstacleStatus")
def get_lidar_obstacle_status():
    """Lấy trạng thái vật cản hiện tại (để debug)"""
    status = shared_data.get_lidar_obstacle()
    return jsonify(status)

@app.route("/getConfig")
@config_login_required
def get_config():
    """Lấy các giá trị cấu hình hiện tại từ runtime_settings"""
    with settings_lock:
        return jsonify(runtime_settings)


@app.route("/listSerialPorts")
@config_login_required
def list_serial_ports():
    return jsonify({
        "success": True,
        "ports": available_serial_ports(),
        "active": ACTIVE_SERIAL_PORTS,
    })


def active_device_test_result(device_type):
    if device_type == "gps_imu":
        status = shared_data.get_gps_imu_status()
        success = status["data_fresh"]
    elif device_type == "controller":
        status = control_link.get_status()
        protocol_identified = status["identified"]
        usb_identified = port_matches_device_usb(
            device_type, ACTIVE_SERIAL_PORTS[device_type]
        )
        success = status["connected"] and (protocol_identified or usb_identified)
    else:
        status = Read_lidar.get_status()
        success = status["connected"] and status["data_fresh"]

    return {
        "success": success,
        "message": (
            f"Đã xác nhận đúng {DEVICE_LABELS[device_type]} trên "
            f"{ACTIVE_SERIAL_PORTS[device_type]}"
            if success
            else (
                f"Cổng {ACTIVE_SERIAL_PORTS[device_type]} đang được ứng dụng sử dụng "
                f"nhưng chưa nhận được dữ liệu đúng của {DEVICE_LABELS[device_type]}"
            )
        ),
        "sample": (
            "Kết nối đang hoạt động"
            if success and device_type != "controller"
            else (
                "USB CH340 đã kết nối; chỉ gửi gói STOP an toàn khi kiểm tra"
                if success
                else ""
            )
        ),
    }


@app.route("/testSerialPort", methods=["POST"])
@config_login_required
def test_serial_port():
    data = request.get_json(silent=True) or {}
    device_type = str(data.get("device_type", "")).strip()
    port = str(data.get("port", "")).strip()

    if device_type not in DEVICE_LABELS:
        return jsonify({"success": False, "message": "Loại thiết bị không hợp lệ"}), 400
    available_names = {item["device"].upper() for item in available_serial_ports()}
    if port.upper() not in available_names:
        return jsonify({"success": False, "message": f"Không tìm thấy cổng {port}"}), 400

    active_owner = next((
        owner for owner, active_port in ACTIVE_SERIAL_PORTS.items()
        if active_port.upper() == port.upper()
    ), None)
    if active_owner is not None and active_owner != device_type:
        result = {
            "success": False,
            "message": (
                f"{port} đang được {DEVICE_LABELS[active_owner]} sử dụng; "
                f"không thể chọn làm {DEVICE_LABELS[device_type]}"
            ),
            "sample": "",
        }
    elif active_owner == device_type:
        result = active_device_test_result(device_type)
    else:
        result = probe_serial_port(device_type, port)

    verified_ports = dict(session.get("verified_serial_ports", {}))
    if result["success"]:
        verified_ports[device_type] = port
    else:
        verified_ports.pop(device_type, None)
    session["verified_serial_ports"] = verified_ports
    return jsonify(result), 200 if result["success"] else 422

@app.route("/saveConfigRuntime", methods=["POST"])
@config_login_required
def save_config_runtime():
    """Lưu cấu hình vào bộ nhớ (runtime) - Áp dụng ngay, mất khi restart"""
    try:
        data = request.get_json()
        print(f"[Settings] 💾 Saving to RUNTIME (temporary): {data}")
        
        values = parse_config_values(data)
        require_verified_serial_changes(values)
        restart_required = serial_ports_restart_required(values)
        apply_config_values(values)
        
        print(f"[Settings] ✅ Runtime settings updated:")
        print(f"  - LIDAR_OBSTACLE_DISTANCE: {get_setting('LIDAR_OBSTACLE_DISTANCE')}mm")
        print(f"  - LIDAR_DETECTION_ANGLE: [{get_setting('LIDAR_DETECTION_ANGLE_MIN')}° to {get_setting('LIDAR_DETECTION_ANGLE_MAX')}°]")
        
        return jsonify({
            "success": True,
            "restart_required": restart_required,
            "message": "Cấu hình đã được lưu vào bộ nhớ (tạm thời)",
        })
    
    except Exception as e:
        print(f"[Settings] ❌ Error saving to runtime: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})

@app.route("/saveConfigPermanent", methods=["POST"])
@config_login_required
def save_config_permanent():
    """Lưu cấu hình vào settings.json - Vĩnh viễn, không cần restart"""
    try:
        data = request.get_json()
        print(f"[Settings] 💿 Saving to settings.json (permanent): {data}")
        
        values = parse_config_values(data)
        require_verified_serial_changes(values)
        restart_required = serial_ports_restart_required(values)
        apply_config_values(values)
        
        # Lưu vào file JSON
        if save_settings_to_json():
            print(f"[Settings] ✅ Settings saved to settings.json permanently")
            return jsonify({
                "success": True,
                "restart_required": restart_required,
                "message": "Cấu hình đã được lưu vĩnh viễn vào settings.json",
            })
        else:
            return jsonify({"success": False, "message": "Lỗi khi ghi file settings.json"})
    
    except Exception as e:
        print(f"[Settings] ❌ Error saving permanent: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})

@app.route("/resetConfig", methods=["POST"])
@config_login_required
def reset_config():
    """Reset về cấu hình mặc định từ config.py"""
    try:
        print("[Settings] 🔄 Resetting to default values from config.py")
        
        # Load lại giá trị mặc định từ config.py
        default_settings = {
            "GPS_IMU_PORT": config.SERIAL_PORT,
            "CONTROL_PORT": config.SERIAL_PORT_CONTROL,
            "LIDAR_PORT": config.LIDAR_PORT,
            "TIME_PER_METER_SEC": config.TIME_PER_METER_SEC,
            "DEFAULT_LATITUDE": config.DEFAULT_LATITUDE,
            "DEFAULT_LONGITUDE": config.DEFAULT_LONGITUDE,
            "LIDAR_MAX_POINTS": config.LIDAR_MAX_POINTS,
            "LIDAR_DOT_LIFETIME": config.LIDAR_DOT_LIFETIME,
            "LIDAR_OBSTACLE_DISTANCE": config.LIDAR_OBSTACLE_DISTANCE,
            "LIDAR_DETECTION_ANGLE_MIN": config.LIDAR_DETECTION_ANGLE_MIN,
            "LIDAR_DETECTION_ANGLE_MAX": config.LIDAR_DETECTION_ANGLE_MAX,
            "IMAGE_2D_ENABLED": config.IMAGE_2D_ENABLED,
            "DETECTION_CONFIDENCE_THRESHOLD": config.DETECTION_CONFIDENCE_THRESHOLD,
            "DETECTION_IOU_THRESHOLD": config.DETECTION_IOU_THRESHOLD,
        }
        
        # Cập nhật runtime_settings
        apply_config_values(default_settings)
        
        # Lưu vào settings.json
        save_settings_to_json()
        
        print("[Settings] ✅ Reset to default values successfully")
        return jsonify({"success": True, "message": "Đã reset về cấu hình mặc định từ config.py"})
    
    except Exception as e:
        print(f"[Settings] ❌ Error resetting: {e}")
        return jsonify({"success": False, "message": str(e)})

if __name__ == "__main__":
    # Khởi động LIDAR
    print("[LIDAR] Auto-starting LIDAR on server startup...")
    Read_lidar.start_lidar_thread(ACTIVE_SERIAL_PORTS["lidar"], LIDAR_BAUDRATE)

    # Khởi động GPIO servo cánh tay
    print("[ARM] Initializing servo GPIO...")
    arm_control.init_servos()

    mdns = MdnsAdvertiser(FLASK_PORT)
    try:
        mdns.start()
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
    finally:
        mdns.stop()
