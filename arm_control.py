"""
arm_control.py
Điều khiển 4 servo cánh tay robot trực tiếp bằng GPIO Raspberry Pi.

Thứ tự ưu tiên thư viện:
  1. pigpio  – hardware-timed PWM, chính xác nhất cho servo
  2. RPi.GPIO – software PWM, fallback khi không có pigpio daemon
  3. Simulation mode – log góc ra console (dùng khi phát triển trên PC)

Pulse width chuẩn servo:
  0°   → 500 μs
  90°  → 1500 μs
  180° → 2500 μs
  Tần số: 50 Hz
"""

import os
import time

# ===================== Cấu hình =====================
try:
    from config import SERVO_J1_PIN, SERVO_J2_PIN, SERVO_J3_PIN, SERVO_J4_PIN
    SERVO_PINS = [SERVO_J1_PIN, SERVO_J2_PIN, SERVO_J3_PIN, SERVO_J4_PIN]
except ImportError:
    # Giá trị mặc định nếu chưa có trong config.py
    SERVO_PINS = [17, 18, 27, 22]

SERVO_FREQ    = 50      # Hz
SERVO_MIN_US  = 500     # μs tại 0°
SERVO_MAX_US  = 2500    # μs tại 180°

# ===================== Phát hiện thư viện =====================
_mode = "simulation"  # "pigpio" | "rpigpio" | "simulation"
_ARM_ENABLED = os.environ.get("ROBOT_ARM_ENABLED", "1").strip().lower() in {
    "1", "true", "yes", "on"
}

if _ARM_ENABLED:
    try:
        import pigpio
        _USE_PIGPIO = True
    except ImportError:
        _USE_PIGPIO = False
else:
    _USE_PIGPIO = False

if not _USE_PIGPIO:
    if _ARM_ENABLED:
        try:
            import RPi.GPIO as GPIO
            _USE_RPIGPIO = True
        except (ImportError, RuntimeError):
            _USE_RPIGPIO = False
    else:
        _USE_RPIGPIO = False
else:
    _USE_RPIGPIO = False

# ===================== State =====================
_pi       = None        # pigpio instance
_pwm_list = []          # RPi.GPIO PWM objects
_initialized = False

# ===================== Helpers =====================

def _angle_to_us(angle: int) -> int:
    """Chuyển góc (0–180) sang pulse width (μs)."""
    angle = max(0, min(180, angle))
    return int(SERVO_MIN_US + (angle / 180.0) * (SERVO_MAX_US - SERVO_MIN_US))

def _angle_to_duty(angle: int) -> float:
    """Chuyển góc sang duty cycle % cho RPi.GPIO (50 Hz)."""
    us       = _angle_to_us(angle)
    period_us = 1_000_000 / SERVO_FREQ
    return (us / period_us) * 100.0

# ===================== Khởi tạo =====================

def init_servos() -> bool:
    """
    Khởi tạo GPIO cho 4 servo.
    Trả về True nếu thành công, False nếu chỉ chạy simulation.
    """
    global _pi, _pwm_list, _mode, _initialized

    if _initialized:
        return _mode != "simulation"

    if not _ARM_ENABLED:
        _mode = "simulation"
        _initialized = True
        print("[ARM] GPIO disabled by ROBOT_ARM_ENABLED=0; simulation mode")
        return False

    if _USE_PIGPIO:
        try:
            _pi = pigpio.pi()
            if not _pi.connected:
                raise RuntimeError("pigpio daemon chưa chạy – hãy chạy 'sudo pigpiod' trước")
            for pin in SERVO_PINS:
                _pi.set_mode(pin, pigpio.OUTPUT)
                _pi.set_servo_pulsewidth(pin, _angle_to_us(90))
            _mode = "pigpio"
            _initialized = True
            print(f"[ARM] pigpio OK – pins BCM {SERVO_PINS}, vị trí 90°")
            return True
        except Exception as e:
            print(f"[ARM] pigpio thất bại: {e}")

    if _USE_RPIGPIO:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            _pwm_list.clear()
            for pin in SERVO_PINS:
                GPIO.setup(pin, GPIO.OUT)
                pwm = GPIO.PWM(pin, SERVO_FREQ)
                pwm.start(_angle_to_duty(90))
                _pwm_list.append(pwm)
            _mode = "rpigpio"
            _initialized = True
            print(f"[ARM] RPi.GPIO (software PWM) OK – pins BCM {SERVO_PINS}")
            return True
        except Exception as e:
            print(f"[ARM] RPi.GPIO thất bại: {e}")

    _mode = "simulation"
    _initialized = True
    print(f"[ARM] Simulation mode (không có GPIO) – pins sẽ được log: {SERVO_PINS}")
    return False

# ===================== Điều khiển =====================

def set_servo_angle(joint_index: int, angle: int):
    """
    Đặt góc cho servo joint_index (0=J1, 1=J2, 2=J3, 3=J4).
    angle: 0–180 độ.
    """
    if joint_index < 0 or joint_index >= 4:
        return
    angle = max(0, min(180, int(angle)))
    pin   = SERVO_PINS[joint_index]

    if _mode == "pigpio" and _pi and _pi.connected:
        _pi.set_servo_pulsewidth(pin, _angle_to_us(angle))

    elif _mode == "rpigpio" and _pwm_list:
        _pwm_list[joint_index].ChangeDutyCycle(_angle_to_duty(angle))

    else:
        # Simulation – chỉ log
        print(f"[ARM-SIM] J{joint_index+1} (GPIO{pin}) = {angle}°")


def set_all_angles(angles: list):
    """
    Đặt đồng thời 4 servo.
    angles: [j1, j2, j3, j4]  – mỗi giá trị 0–180 độ.
    """
    if not _initialized:
        init_servos()
    for i, angle in enumerate(angles[:4]):
        set_servo_angle(i, int(angle))
    print(f"[ARM] Angles set: J1={angles[0]}° J2={angles[1]}° J3={angles[2]}° J4={angles[3]}°")


def get_mode() -> str:
    """Trả về mode hiện tại: 'pigpio' | 'rpigpio' | 'simulation'."""
    return _mode

# ===================== Dọn dẹp =====================

def cleanup():
    """Giải phóng tài nguyên GPIO (gọi khi tắt server)."""
    global _pi, _pwm_list, _initialized
    if _mode == "pigpio" and _pi:
        for pin in SERVO_PINS:
            _pi.set_servo_pulsewidth(pin, 0)   # tắt xung để servo không rung
        _pi.stop()
        _pi = None
        print("[ARM] pigpio cleanup done")
    elif _mode == "rpigpio" and _pwm_list:
        for pwm in _pwm_list:
            pwm.stop()
        GPIO.cleanup()
        _pwm_list.clear()
        print("[ARM] RPi.GPIO cleanup done")
    _initialized = False
