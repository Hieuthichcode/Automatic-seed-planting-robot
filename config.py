# Serial cấu hình MPU và GPS
SERIAL_PORT = "COM43"   # hoặc /dev/ttyACM0 nếu ESP32-C3 hiển thị vậy
SERIAL_BAUD = 115200
# Cổng COM dùng để gửi packet điều khiển đến ESP32 chính
SERIAL_PORT_CONTROL = "COM48"  # chỉnh lại theo thực tế
SERIAL_BAUD_CONTROL = 115200  # baudrate cho cổng điều khiển

# Giao thức điều khiển Raspberry Pi -> ESP32 chính
CONTROL_HEARTBEAT_HZ = 25
CONTROL_WRITE_TIMEOUT_SEC = 0.05
CONTROL_RECONNECT_DELAY_SEC = 2.0
CONTROL_STARTUP_STOP_FRAMES = 5
CONTROL_STOP_PULSE_SEC = 0.5
AUTO_DRIVE_ENABLED = True
# Giá trị packet chuẩn hóa trong miền 0..1000; ESP32 đổi sang PWM 0..255.
AUTO_FORWARD_COMMAND = 700
AUTO_TURN_COMMAND = 600
# Giữ ít nhất ba heartbeat STOP khi đổi giữa đi thẳng và quay tại chỗ.
AUTO_TRANSITION_STOP_SEC = 0.12
# Thời gian tối đa đang phát lệnh quay cho một đoạn; quá hạn thì hủy route.
AUTO_TURN_TIMEOUT_SEC = 15.0
AUTO_MOVE_STOP = 0
AUTO_MOVE_FORWARD = 1
AUTO_MOVE_PIVOT_LEFT = 2
AUTO_MOVE_PIVOT_RIGHT = 3

# # Serial cấu hình
# SERIAL_PORT = "/dev/ttyACM1"   # hoặc /dev/ttyACM0 nếu ESP32-C3 hiển thị vậy
# SERIAL_BAUD = 115200
# # Cổng COM dùng để gửi lệnh điều khiển (khác cổng đọc ESP)
# SERIAL_PORT_CONTROL = "/dev/ttyUSB0"  # chỉnh lại theo thực tế
# SERIAL_BAUD_CONTROL = 115200  # baudrate cho cổng điều khiển



# Flask cấu hình
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 80      # cổng HTTP mặc định: truy cập http://control.local

# Cho phép tải ảnh từ máy tính để chạy cùng pipeline detection với camera.
IMAGE_2D_ENABLED = False
DETECTION_CONFIDENCE_THRESHOLD = 0.30
DETECTION_IOU_THRESHOLD = 0.45

# Google Maps
GOOGLE_MAPS_API_KEY = "AIzaSyCWxuTGtkD5O7cDqQYrsHZeajpPBZdKWnk"  # thay bằng key thật

# Timeout Serial
SERIAL_TIMEOUT_SEC = 1

# Thời gian trung bình xe đi hết 1 mét (giây)
TIME_PER_METER_SEC = 1.0  # chỉnh lại theo thực tế

# Tọa độ dùng khi GPS chưa có tín hiệu hợp lệ
DEFAULT_LATITUDE = 20.962059
DEFAULT_LONGITUDE = 105.748392

# =====================================================
# LIDAR Configuration
# =====================================================
# Cấu hình Serial cho LIDAR
#LIDAR_PORT = "/dev/ttyACM0"           # Cổng COM kết nối LIDAR
LIDAR_PORT = "COM44"           # Cổng COM kết nối LIDAR
LIDAR_BAUDRATE = 115200       # Tốc độ baudrate
LIDAR_SERIAL_TIMEOUT_SEC = 0.1
LIDAR_RECONNECT_DELAY_SEC = 2.0
LIDAR_DATA_TIMEOUT_SEC = 0.6

# Cấu hình hiển thị LIDAR
LIDAR_MAX_POINTS = 1500       # Số điểm tối đa lưu trữ
LIDAR_DOT_LIFETIME = 1.0      # Thời gian hiển thị điểm (giây)

# Cấu hình phát hiện vật cản
LIDAR_OBSTACLE_DISTANCE = 400  # Khoảng cách phát hiện vật cản (mm)
LIDAR_DETECTION_ANGLE_MIN = -20  # Góc phát hiện tối thiểu (độ)
LIDAR_DETECTION_ANGLE_MAX = 20   # Góc phát hiện tối đa (độ)

# =====================================================
# Servo Cánh tay Robot – GPIO BCM pins (Raspberry Pi)
# =====================================================
SERVO_J1_PIN = 17   # GPIO17 (pin 11) – Khớp gốc
SERVO_J2_PIN = 18   # GPIO18 (pin 12) – Khớp vai
SERVO_J3_PIN = 27   # GPIO27 (pin 13) – Khớp khuỷu
SERVO_J4_PIN = 22   # GPIO22 (pin 15) – Khớp cổ tay


