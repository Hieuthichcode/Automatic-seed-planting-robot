import math
import serial
import threading
import time
from time import sleep
from config import (
    AUTO_MOVE_FORWARD,
    AUTO_MOVE_PIVOT_LEFT,
    AUTO_MOVE_PIVOT_RIGHT,
    AUTO_MOVE_STOP,
    AUTO_TRANSITION_STOP_SEC,
    AUTO_TURN_TIMEOUT_SEC,
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


def validate_route_commands(dis_list, dir_list, dir_value_list):
    """Validate and normalize the three parallel route arrays."""
    if not dis_list:
        raise ValueError("Hành trình phải có ít nhất một đoạn đường")
    if not (len(dis_list) == len(dir_list) == len(dir_value_list)):
        raise ValueError("Ba danh sách quãng đường, hướng và góc phải cùng độ dài")

    distances = []
    directions = []
    angles = []
    for index, (distance, direction, angle) in enumerate(
            zip(dis_list, dir_list, dir_value_list), start=1):
        distance = float(distance)
        direction = int(direction)
        angle = float(angle)
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError(f"Quãng đường đoạn {index} phải lớn hơn 0")
        if direction not in (-1, 0, 1):
            raise ValueError(f"Hướng đoạn {index} chỉ được là -1, 0 hoặc 1")
        if not math.isfinite(angle) or not 0 <= angle <= 180:
            raise ValueError(f"Góc đoạn {index} phải nằm trong khoảng 0..180 độ")
        if direction != 0 and angle <= 0:
            raise ValueError(f"Đoạn {index} có hướng quay nhưng góc bằng 0")
        distances.append(distance)
        directions.append(direction)
        angles.append(angle)
    return distances, directions, angles

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
        self.move_cmd = AUTO_MOVE_STOP
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

    def try_begin_route(self, total_steps):
        """Atomically reserve route execution so two workers cannot overlap."""
        with self.lock:
            if self.route_running:
                return False
            self.route_running = True
            self.route_paused = False
            self.route_stopped = False
            self.current_step = 0
            self.total_steps = int(total_steps)
            self.current_action = "Bắt đầu hành trình..."
            self.distance_remaining = 0.0
            self.move_cmd = AUTO_MOVE_STOP
            return True
    
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
            self.move_cmd = AUTO_MOVE_STOP
            self.route_running = False
            self.route_paused = False
            self.route_stopped = True

    def set_move_cmd(self, cmd):
        """Set AUTO move command: stop, forward, pivot-left or pivot-right."""
        with self.lock:
            self.move_cmd = int(cmd)

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
def _lidar_blocks_motion(status):
    return not status.get("valid", False) or bool(status.get("detected"))


def _stop_for_transition(shared_data):
    """Publish STOP long enough for several heartbeat frames before reversing."""
    shared_data.set_move_cmd(AUTO_MOVE_STOP)
    deadline = time.monotonic() + AUTO_TRANSITION_STOP_SEC
    while time.monotonic() < deadline:
        if shared_data.get_route_state()["stopped"]:
            return False
        sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    return True


def execute_route_commands(
        shared_data: SerialData, dis_list, dir_list, dir_value_list,
        route_prepared=False):
    """
    Thực thi tuần tự các lệnh điều khiển xe.
    Thay vì ghi serial trực tiếp, cập nhật shared_data.move_cmd.
    Luồng gửi duy nhất sẽ đọc trạng thái này và phát packet AUTO định kỳ.
    """
    try:
        dis_list, dir_list, dir_value_list = validate_route_commands(
            dis_list,
            dir_list,
            dir_value_list,
        )
        if not route_prepared and not shared_data.try_begin_route(len(dis_list)):
            print("[Serial] Route already running; duplicate worker ignored")
            return

        print("[Serial] === Route execution started ===")
        shared_data.update_route_progress(0, len(dis_list), "Bắt đầu hành trình...")
        shared_data.set_move_cmd(AUTO_MOVE_STOP)

        for i in range(len(dis_list)):
            # Kiểm tra nếu bị dừng
            if shared_data.get_route_state()["stopped"]:
                print("[Serial] Route stopped by user")
                shared_data.set_move_cmd(AUTO_MOVE_STOP)
                break

            distance = float(dis_list[i])
            direction = int(dir_list[i])
            angle = float(dir_value_list[i])
            print(f"[Serial] Step {i+1}/{len(dis_list)}: dis={distance:.2f}, dir={direction}, angle={angle:.2f}")

            # Bước 1: Đi tiến
            shared_data.update_route_progress(i+1, len(dis_list), "Đi thẳng", distance)
            print(f"[Serial] move_cmd=1 (forward {distance:.2f}m)")
            total_time = TIME_PER_METER_SEC * distance
            elapsed = 0
            interval = 0.1

            while elapsed < total_time:
                while shared_data.get_route_state()["paused"]:
                    shared_data.set_move_cmd(AUTO_MOVE_STOP)
                    shared_data.update_route_progress(i+1, len(dis_list), "Đang tạm dừng...", distance * (1 - elapsed/total_time))
                    if shared_data.get_route_state()["stopped"]:
                        return
                    sleep(0.1)

                if shared_data.get_route_state()["stopped"]:
                    shared_data.set_move_cmd(AUTO_MOVE_STOP)
                    shared_data.update_route_progress(i+1, len(dis_list), "Đã dừng")
                    print("[Serial] Route stopped during movement")
                    return

                # Lidar obstacle hoặc dữ liệu Lidar stale đều giữ STOP và không
                # cộng thời gian di chuyển.
                lidar_status = shared_data.get_lidar_obstacle()
                if _lidar_blocks_motion(lidar_status):
                    shared_data.set_move_cmd(AUTO_MOVE_STOP)
                    if lidar_status.get("valid", False):
                        hold_message = (
                            f"Vật cản {lidar_status['min_distance']:.0f}mm - Tạm dừng"
                        )
                    else:
                        hold_message = "Mất dữ liệu LIDAR - Tạm dừng"
                    print(f"[Serial] {hold_message}")
                    shared_data.update_route_progress(
                        i+1,
                        len(dis_list),
                        hold_message,
                        distance * (1 - elapsed/total_time),
                    )
                    while _lidar_blocks_motion(lidar_status):
                        if shared_data.get_route_state()["stopped"]:
                            shared_data.set_move_cmd(AUTO_MOVE_STOP)
                            print("[Serial] Route stopped during LIDAR wait")
                            return
                        sleep(0.1)
                        lidar_status = shared_data.get_lidar_obstacle()
                    print("[Serial] LIDAR clear — Resuming movement")
                    shared_data.update_route_progress(
                        i+1,
                        len(dis_list),
                        "Tiếp tục di chuyển",
                        distance * (1 - elapsed/total_time),
                    )
                    continue

                shared_data.set_move_cmd(AUTO_MOVE_FORWARD)
                remaining_distance = distance * (1 - elapsed/total_time)
                shared_data.update_route_progress(i+1, len(dis_list), "Đi thẳng", remaining_distance)
                sleep(interval)
                elapsed += interval

            shared_data.update_route_progress(i+1, len(dis_list), f"Đã đi xong {distance:.2f}m", 0)

            # Bước 2: Quay nếu cần
            if direction != 0:
                if shared_data.get_route_state()["stopped"]:
                    shared_data.set_move_cmd(AUTO_MOVE_STOP)
                    shared_data.update_route_progress(i+1, len(dis_list), "Đã dừng")
                    return

                turn_direction = "trái" if direction == -1 else "phải"
                move_code = (
                    AUTO_MOVE_PIVOT_LEFT
                    if direction == -1
                    else AUTO_MOVE_PIVOT_RIGHT
                )
                shared_data.update_route_progress(i+1, len(dis_list), f"Quay {turn_direction} {angle:.2f}°")
                if not _stop_for_transition(shared_data):
                    return

                if not shared_data.imu_is_fresh():
                    shared_data.set_move_cmd(AUTO_MOVE_STOP)
                    shared_data.set_route_state(running=False, stopped=True)
                    shared_data.update_route_progress(i+1, len(dis_list), "Mất dữ liệu IMU")
                    print("[Serial] IMU data is stale before turn; route stopped")
                    return

                start_yaw = shared_data.snapshot().get("yaw", 0.0)
                target_yaw = start_yaw + angle if direction == -1 else start_yaw - angle
                target_yaw = ((target_yaw + 180.0) % 360.0) - 180.0
                last_yaw = start_yaw
                turned_angle = 0.0
                turn_active_sec = 0.0
                print(f"[Serial] Start yaw: {start_yaw:.2f}°, Target: {target_yaw:.2f}°, Turn: {angle:.2f}°")

                while True:
                    # Kiểm tra pause
                    while shared_data.get_route_state()["paused"]:
                        shared_data.set_move_cmd(AUTO_MOVE_STOP)
                        shared_data.update_route_progress(i+1, len(dis_list), "Đang tạm dừng...")
                        if shared_data.get_route_state()["stopped"]:
                            return
                        sleep(0.1)

                    if shared_data.get_route_state()["stopped"]:
                        shared_data.set_move_cmd(AUTO_MOVE_STOP)
                        shared_data.update_route_progress(i+1, len(dis_list), "Đã dừng")
                        return

                    lidar_status = shared_data.get_lidar_obstacle()
                    if _lidar_blocks_motion(lidar_status):
                        shared_data.set_move_cmd(AUTO_MOVE_STOP)
                        hold_message = (
                            f"Vật cản {lidar_status['min_distance']:.0f}mm - Tạm dừng"
                            if lidar_status.get("valid", False)
                            else "Mất dữ liệu LIDAR - Tạm dừng"
                        )
                        shared_data.update_route_progress(
                            i+1, len(dis_list), hold_message, 0
                        )
                        while _lidar_blocks_motion(lidar_status):
                            if shared_data.get_route_state()["stopped"]:
                                shared_data.set_move_cmd(AUTO_MOVE_STOP)
                                print("[Serial] Route stopped during LIDAR wait (turn)")
                                return
                            sleep(0.1)
                            lidar_status = shared_data.get_lidar_obstacle()
                        print("[Serial] LIDAR clear — Resuming turn")
                        shared_data.update_route_progress(
                            i+1,
                            len(dis_list),
                            f"Tiếp tục quay {turn_direction}",
                            0,
                        )
                        continue

                    if not shared_data.imu_is_fresh():
                        shared_data.set_move_cmd(AUTO_MOVE_STOP)
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
                        if not _stop_for_transition(shared_data):
                            return
                        break

                    turn_active_sec += 0.05
                    if turn_active_sec > AUTO_TURN_TIMEOUT_SEC:
                        shared_data.set_move_cmd(AUTO_MOVE_STOP)
                        shared_data.set_route_state(running=False, stopped=True)
                        shared_data.update_route_progress(
                            i+1, len(dis_list), "Quay quá thời gian cho phép")
                        print("[Serial] Turn timeout (yaw not changing?); route stopped")
                        return

                    shared_data.set_move_cmd(move_code)
                    sleep(0.05)
            else:
                print("[Serial] No turn needed")

        # Kết thúc hành trình
        shared_data.set_move_cmd(AUTO_MOVE_STOP)
        shared_data.set_route_state(running=False, paused=False, stopped=False)
        shared_data.update_route_progress(len(dis_list), len(dis_list), "Hoàn thành hành trình!")
        print("[Serial] === Route execution finished ===")

    except Exception as e:
        print(f"[Serial] Route execution error: {e}")
        shared_data.set_move_cmd(AUTO_MOVE_STOP)
        shared_data.set_route_state(running=False, stopped=True)
        shared_data.update_route_progress(0, 0, f"Lỗi: {str(e)}")
