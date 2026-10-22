import math
import serial
import threading
import time
from time import sleep
from config import (
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    SERIAL_BAUD,
    SERIAL_PORT,
    SERIAL_TIMEOUT_SEC,
    TIME_PER_METER_SEC,
)


GPS_IMU_FIELD_COUNT = 20
IMU_FRESH_TIMEOUT_SEC = 0.5


def parse_gps_imu_line(line):
    """Parse the 20-field CSV emitted by CODE/GPS_MPU/GPS_MPU.ino."""
    parts = line.split(",")
    if len(parts) != GPS_IMU_FIELD_COUNT:
        return None

    try:
        source_ms = int(parts[0])
        yaw = float(parts[1])
        gps_fix = int(parts[11])
        latitude = float(parts[12])
        longitude = float(parts[13])
        speed_kmph = float(parts[15])
        gps_age_ms = int(parts[19])
    except (ValueError, IndexError):
        return None

    if not math.isfinite(yaw) or gps_fix not in (0, 1):
        return None

    data = {
        "yaw": ((yaw + 180.0) % 360.0) - 180.0,
        "source_ms": source_ms,
        "gps_valid": False,
        "gps_speed_kmph": speed_kmph if math.isfinite(speed_kmph) else 0.0,
        "gps_age_ms": gps_age_ms,
    }

    location_valid = (
        gps_fix == 1
        and math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90.0 <= latitude <= 90.0
        and -180.0 <= longitude <= 180.0
        and gps_age_ms >= 0
    )
    if location_valid:
        data.update({
            "lat": latitude,
            "lon": longitude,
            "gps_valid": True,
        })
    return data


def signed_angle_delta(current, previous):
    """Shortest signed angular delta in degrees, safe across +/-180."""
    return ((current - previous + 180.0) % 360.0) - 180.0

class SerialData:
    """Bộ nhớ chia sẻ để lưu dữ liệu GPS mới nhất từ ESP32"""
    def __init__(self):
        self.latest = {
            "lat": DEFAULT_LATITUDE,
            "lon": DEFAULT_LONGITUDE,
            "yaw": 0.0,
            "gps_valid": False,
            "gps_speed_kmph": 0.0,
            "gps_age_ms": -1,
            "source_ms": 0,
        }
        self.lock = threading.Lock()
        self.gps_imu_connected = False
        self.last_imu_rx_monotonic = 0.0
        # Thêm biến điều khiển route
        self.route_running = False
        self.route_paused = False
        self.route_stopped = False
        self.current_step = 0
        self.total_steps = 0
        self.current_action = "Chờ lệnh..."
        self.distance_remaining = 0.0
        # Thêm biến LIDAR obstacle detection
        self.lidar_obstacle_detected = False
        self.lidar_min_distance = float('inf')
        self.lidar_data_valid = False
        # Lệnh di chuyển và góc servo — giao tiếp với unified sender thread
        self.move_cmd = 0            # 0=dừng, 1=tiến, 2=trái, 3=phải
        self.arm_angles = [90, 90, 90, 90]  # J1,J2,J3,J4

    def update(self, data: dict):
        with self.lock:
            self.latest.update(data)

    def update_gps_imu(self, data: dict):
        with self.lock:
            self.latest.update(data)
            self.gps_imu_connected = True
            self.last_imu_rx_monotonic = time.monotonic()

    def mark_gps_imu_disconnected(self):
        with self.lock:
            self.gps_imu_connected = False
            self.latest["gps_valid"] = False

    def imu_is_fresh(self, max_age=IMU_FRESH_TIMEOUT_SEC):
        with self.lock:
            return (
                self.gps_imu_connected
                and self.last_imu_rx_monotonic > 0
                and time.monotonic() - self.last_imu_rx_monotonic <= max_age
            )

    def snapshot(self):
        with self.lock:
            return dict(self.latest)

    def get_gps_imu_status(self):
        with self.lock:
            fresh = (
                self.gps_imu_connected
                and self.last_imu_rx_monotonic > 0
                and time.monotonic() - self.last_imu_rx_monotonic <= IMU_FRESH_TIMEOUT_SEC
            )
            return {
                "connected": self.gps_imu_connected,
                "data_fresh": fresh,
            }
    
    def set_route_state(self, running=None, paused=None, stopped=None):
        with self.lock:
            if running is not None:
                self.route_running = running
            if paused is not None:
                self.route_paused = paused
            if stopped is not None:
                self.route_stopped = stopped
    
    def get_route_state(self):
        with self.lock:
            return {
                "running": self.route_running,
                "paused": self.route_paused,
                "stopped": self.route_stopped,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "current_action": self.current_action,
                "distance_remaining": self.distance_remaining
            }
    
    def update_route_progress(self, step, total, action, distance_remaining=0.0):
        with self.lock:
            self.current_step = step
            self.total_steps = total
            self.current_action = action
            self.distance_remaining = distance_remaining
    
    def set_lidar_obstacle(self, detected, min_distance=float('inf'), valid=True):
        """Cập nhật trạng thái phát hiện vật cản từ LIDAR"""
        with self.lock:
            self.lidar_obstacle_detected = detected
            self.lidar_min_distance = min_distance
            self.lidar_data_valid = valid
    
    def get_lidar_obstacle(self):
        """Lấy trạng thái phát hiện vật cản"""
        with self.lock:
            return {
                "detected": self.lidar_obstacle_detected,
                "min_distance": self.lidar_min_distance,
                "valid": self.lidar_data_valid,
            }

    def get_control_snapshot(self):
        """Atomic snapshot consumed by the single ESP32 sender thread."""
        with self.lock:
            return {
                "route_running": self.route_running and not self.route_stopped,
                "move_cmd": self.move_cmd,
                "lidar_detected": self.lidar_obstacle_detected,
                "lidar_min_distance": self.lidar_min_distance,
                "lidar_valid": self.lidar_data_valid,
            }

    def invalidate_motion_for_link_loss(self):
        """Cancel the current route so reconnect cannot replay an old move."""
        with self.lock:
            self.move_cmd = 0
            self.route_running = False
            self.route_paused = False
            self.route_stopped = True

    def set_move_cmd(self, cmd):
        """Đặt lệnh di chuyển (0=dừng, 1=tiến, 2=trái, 3=phải)"""
        with self.lock:
            self.move_cmd = cmd

    def get_move_cmd(self):
        """Lấy lệnh di chuyển hiện tại"""
        with self.lock:
            return self.move_cmd

    def set_arm_angles(self, angles):
        """Cập nhật 4 góc servo cánh tay [J1, J2, J3, J4]"""
        with self.lock:
            self.arm_angles = [int(a) for a in angles]

    def get_arm_angles(self):
        """Lấy 4 góc servo cánh tay"""
        with self.lock:
            return list(self.arm_angles)

def start_serial_thread(shared_data: SerialData, port=SERIAL_PORT):
    """Luồng đọc dữ liệu GPS và góc quay từ ESP32"""
    while True:
        try:
            with serial.Serial(port, SERIAL_BAUD, timeout=SERIAL_TIMEOUT_SEC) as ser:
                print(f"[GPS/IMU] Connected to {port} @ {SERIAL_BAUD}")
                while True:
                    line = ser.readline().decode(errors="ignore").strip()
                    if not line:
                        continue
                    data = parse_gps_imu_line(line)
                    if data is not None:
                        shared_data.update_gps_imu(data)
        except Exception as e:
            shared_data.mark_gps_imu_disconnected()
            print(f"[GPS/IMU] Error: {e}. Reconnecting in 2s...")
            sleep(2)
def execute_route_commands(shared_data: SerialData, dis_list, dir_list, dir_value_list):
    """
    Thực thi tuần tự các lệnh điều khiển xe.
    Thay vì ghi serial trực tiếp, cập nhật shared_data.move_cmd.
    Luồng gửi duy nhất sẽ đọc trạng thái này; AUTO vẫn bị khóa an toàn (TODO).
    """
    try:
        print("[Serial] === Route execution started ===")
        shared_data.set_route_state(running=True, paused=False, stopped=False)
        shared_data.update_route_progress(0, len(dis_list), "Bắt đầu hành trình...")
        shared_data.set_move_cmd(0)

        for i in range(len(dis_list)):
            # Kiểm tra nếu bị dừng
            if shared_data.get_route_state()["stopped"]:
                print("[Serial] Route stopped by user")
                shared_data.set_move_cmd(0)
                break

            distance = float(dis_list[i])
            direction = int(dir_list[i])
            angle = float(dir_value_list[i])
            print(f"[Serial] Step {i+1}/{len(dis_list)}: dis={distance:.2f}, dir={direction}, angle={angle:.2f}")

            # Bước 1: Đi tiến
            shared_data.update_route_progress(i+1, len(dis_list), "Đi thẳng", distance)
            shared_data.set_move_cmd(1)  # 1 = tiến
            print(f"[Serial] move_cmd=1 (forward {distance:.2f}m)")
            total_time = TIME_PER_METER_SEC * distance
            elapsed = 0
            interval = 0.1

            while elapsed < total_time:
                # Kiểm tra pause
                while shared_data.get_route_state()["paused"]:
                    shared_data.set_move_cmd(0)
                    shared_data.update_route_progress(i+1, len(dis_list), "Đang tạm dừng...", distance * (1 - elapsed/total_time))
                    sleep(0.1)
                shared_data.set_move_cmd(1)  # Tiếp tục tiến sau khi hết pause

                # Kiểm tra LIDAR obstacle — dừng đếm thời gian để không mất quãng đường
                lidar_status = shared_data.get_lidar_obstacle()
                if lidar_status["detected"]:
                    print(f"[Serial] ⚠️ LIDAR OBSTACLE! {lidar_status['min_distance']:.0f}mm — pausing timer")
                    shared_data.update_route_progress(i+1, len(dis_list),
                        f"⚠️ Vật cản {lidar_status['min_distance']:.0f}mm - Tạm dừng",
                        distance * (1 - elapsed/total_time))
                    # Sender packet đặt cờ OBSTACLE/STOP; timer route chờ thông đường.
                    while lidar_status["detected"]:
                        if shared_data.get_route_state()["stopped"]:
                            shared_data.set_move_cmd(0)
                            print("[Serial] Route stopped during LIDAR obstacle wait")
                            return
                        sleep(0.1)
                        lidar_status = shared_data.get_lidar_obstacle()
                    print("[Serial] LIDAR obstacle cleared — Resuming movement")
                    shared_data.update_route_progress(i+1, len(dis_list), "Tiếp tục di chuyển", distance * (1 - elapsed/total_time))
                    shared_data.set_move_cmd(1)

                # Kiểm tra stop
                if shared_data.get_route_state()["stopped"]:
                    shared_data.set_move_cmd(0)
                    shared_data.update_route_progress(i+1, len(dis_list), "Đã dừng")
                    print("[Serial] Route stopped during movement")
                    return

                remaining_distance = distance * (1 - elapsed/total_time)
                shared_data.update_route_progress(i+1, len(dis_list), "Đi thẳng", remaining_distance)
                sleep(interval)
                elapsed += interval

            shared_data.update_route_progress(i+1, len(dis_list), f"Đã đi xong {distance:.2f}m", 0)

            # Bước 2: Quay nếu cần
            if direction != 0:
                if shared_data.get_route_state()["stopped"]:
                    shared_data.set_move_cmd(0)
                    shared_data.update_route_progress(i+1, len(dis_list), "Đã dừng")
                    return

                turn_direction = "trái" if direction == -1 else "phải"
                move_code = 2 if direction == -1 else 3  # 2=trái, 3=phải
                shared_data.update_route_progress(i+1, len(dis_list), f"Quay {turn_direction} {angle:.2f}°")
                shared_data.set_move_cmd(0)  # Dừng ngắn trước khi quay

                if not shared_data.imu_is_fresh():
                    shared_data.set_move_cmd(0)
                    shared_data.set_route_state(running=False, stopped=True)
                    shared_data.update_route_progress(i+1, len(dis_list), "Mất dữ liệu IMU")
                    print("[Serial] IMU data is stale before turn; route stopped")
                    return

                start_yaw = shared_data.snapshot().get("yaw", 0.0)
                target_yaw = start_yaw + angle if direction == -1 else start_yaw - angle
                target_yaw = ((target_yaw + 180.0) % 360.0) - 180.0
                last_yaw = start_yaw
                turned_angle = 0.0
                print(f"[Serial] Start yaw: {start_yaw:.2f}°, Target: {target_yaw:.2f}°, Turn: {angle:.2f}°")

                while True:
                    # Kiểm tra pause
                    while shared_data.get_route_state()["paused"]:
                        shared_data.set_move_cmd(0)
                        shared_data.update_route_progress(i+1, len(dis_list), "Đang tạm dừng...")
                        sleep(0.1)

                    # Kiểm tra LIDAR trong lúc quay
                    lidar_status = shared_data.get_lidar_obstacle()
                    if lidar_status["detected"]:
                        print(f"[Serial] ⚠️ LIDAR during turn! {lidar_status['min_distance']:.0f}mm")
                        shared_data.update_route_progress(i+1, len(dis_list),
                            f"⚠️ Vật cản {lidar_status['min_distance']:.0f}mm - Tạm dừng", 0)
                        while lidar_status["detected"]:
                            if shared_data.get_route_state()["stopped"]:
                                shared_data.set_move_cmd(0)
                                print("[Serial] Route stopped during LIDAR obstacle wait (turn)")
                                return
                            sleep(0.1)
                            lidar_status = shared_data.get_lidar_obstacle()
                        print("[Serial] LIDAR obstacle cleared — Resuming turn")
                        shared_data.update_route_progress(i+1, len(dis_list), f"Tiếp tục quay {turn_direction}", 0)

                    # Kiểm tra stop
                    if shared_data.get_route_state()["stopped"]:
                        shared_data.set_move_cmd(0)
                        shared_data.update_route_progress(i+1, len(dis_list), "Đã dừng")
                        return

                    if not shared_data.imu_is_fresh():
                        shared_data.set_move_cmd(0)
                        shared_data.set_route_state(running=False, stopped=True)
                        shared_data.update_route_progress(i+1, len(dis_list), "Mất dữ liệu IMU")
                        print("[Serial] IMU data became stale during turn; route stopped")
                        return

                    current_yaw = shared_data.snapshot().get("yaw", 0.0)
                    yaw_step = signed_angle_delta(current_yaw, last_yaw)
                    directed_step = yaw_step if direction == -1 else -yaw_step
                    if directed_step > 0:
                        turned_angle += directed_step
                    last_yaw = current_yaw

                    remaining_angle = max(0.0, angle - turned_angle)
                    shared_data.update_route_progress(i+1, len(dis_list), f"Quay {turn_direction} (còn {remaining_angle:.1f}°)")
                    print(f"[Serial] Yaw: {current_yaw:.2f}°, Turned: {turned_angle:.2f}°/{angle:.2f}°")

                    if turned_angle >= angle:
                        print(f"[Serial] Done turn {angle:.2f}°")
                        shared_data.update_route_progress(i+1, len(dis_list), f"Đã quay xong {angle:.2f}°")
                        shared_data.set_move_cmd(0)
                        break

                    shared_data.set_move_cmd(move_code)
                    sleep(0.05)
            else:
                print("[Serial] No turn needed")

        # Kết thúc hành trình
        shared_data.set_move_cmd(0)
        shared_data.set_route_state(running=False)
        shared_data.update_route_progress(len(dis_list), len(dis_list), "Hoàn thành hành trình!")
        print("[Serial] === Route execution finished ===")

    except Exception as e:
        print(f"[Serial] Route execution error: {e}")
        shared_data.set_move_cmd(0)
        shared_data.set_route_state(running=False, stopped=True)
        shared_data.update_route_progress(0, 0, f"Lỗi: {str(e)}")
