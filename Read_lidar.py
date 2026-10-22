import math
import threading
import time

import serial
import serial.tools.list_ports

from config import (
    LIDAR_DATA_TIMEOUT_SEC,
    LIDAR_DOT_LIFETIME,
    LIDAR_MAX_POINTS,
    LIDAR_RECONNECT_DELAY_SEC,
    LIDAR_SERIAL_TIMEOUT_SEC,
)


class LidarData:
    """Thread-safe store for recent Lidar scan points."""

    def __init__(self):
        self.lock = threading.Lock()
        self.angles = []
        self.distances = []
        self.qualities = []
        self.timestamps = []
        self.max_points = LIDAR_MAX_POINTS
        self.dot_lifetime = LIDAR_DOT_LIFETIME

    def add_point(self, angle_deg, distance_mm, quality=0.0):
        with self.lock:
            self.angles.append(math.radians(angle_deg % 360.0))
            self.distances.append(distance_mm)
            self.qualities.append(quality)
            self.timestamps.append(time.monotonic())

            if len(self.angles) > self.max_points:
                self.angles.pop(0)
                self.distances.pop(0)
                self.qualities.pop(0)
                self.timestamps.pop(0)

    def get_current_points(self):
        with self.lock:
            now = time.monotonic()
            valid_points = []
            for index in range(len(self.angles)):
                age = now - self.timestamps[index]
                if age < self.dot_lifetime:
                    valid_points.append({
                        "angle": math.degrees(self.angles[index]),
                        "distance": self.distances[index],
                        "quality": self.qualities[index],
                        "age": age,
                    })
            return valid_points

    def clear_all(self):
        with self.lock:
            self.angles.clear()
            self.distances.clear()
            self.qualities.clear()
            self.timestamps.clear()


def parse_lidar_line(line):
    """Parse ``distance angle [quality]`` emitted by the Lidar ESP32."""
    parts = line.replace(",", " ").replace(";", " ").split()
    if len(parts) not in (2, 3):
        return None

    try:
        distance = float(parts[0])
        angle = float(parts[1])
        quality = float(parts[2]) if len(parts) == 3 else 0.0
    except ValueError:
        return None

    if not all(math.isfinite(value) for value in (distance, angle, quality)):
        return None
    if distance <= 0 or quality < 0:
        return None
    return distance, angle % 360.0, quality


lidar_data = LidarData()
ser = None
connected = False
last_valid_point_monotonic = 0.0

_state_lock = threading.Lock()
_start_lock = threading.Lock()
_stop_event = threading.Event()
_reader_thread = None
_last_error_text = ""
_last_error_log_monotonic = 0.0


def _set_connection(serial_port, is_connected):
    global ser, connected
    with _state_lock:
        ser = serial_port
        connected = is_connected


def read_lidar_serial(port, baudrate):
    """Own the Lidar port and reconnect until ``stop_lidar`` is called."""
    global last_valid_point_monotonic
    global _last_error_text, _last_error_log_monotonic

    while not _stop_event.is_set():
        serial_port = None
        try:
            serial_port = serial.Serial(
                port,
                baudrate,
                timeout=LIDAR_SERIAL_TIMEOUT_SEC,
            )
            _set_connection(serial_port, True)
            _last_error_text = ""
            print(f"[LIDAR] Connected to {port} @ {baudrate}")

            while not _stop_event.is_set():
                raw = serial_port.readline()
                if not raw:
                    continue
                parsed = parse_lidar_line(raw.decode(errors="ignore").strip())
                if parsed is None:
                    continue

                distance, angle, quality = parsed
                lidar_data.add_point(angle, distance, quality)
                with _state_lock:
                    last_valid_point_monotonic = time.monotonic()

        except Exception as exc:
            if not _stop_event.is_set():
                message = str(exc)
                now = time.monotonic()
                if (
                    message != _last_error_text
                    or now - _last_error_log_monotonic >= 30.0
                ):
                    print(f"[LIDAR] Connection/read error: {message}")
                    _last_error_text = message
                    _last_error_log_monotonic = now
        finally:
            if serial_port is not None:
                try:
                    serial_port.close()
                except Exception:
                    pass
            _set_connection(None, False)
            with _state_lock:
                last_valid_point_monotonic = 0.0

        if not _stop_event.is_set():
            _stop_event.wait(LIDAR_RECONNECT_DELAY_SEC)

    print("[LIDAR] Serial reader stopped")


def start_lidar_thread(port="COM3", baudrate=115200):
    global _reader_thread, last_valid_point_monotonic

    with _start_lock:
        if _reader_thread is not None and _reader_thread.is_alive():
            return False
        _stop_event.clear()
        with _state_lock:
            last_valid_point_monotonic = 0.0
        _reader_thread = threading.Thread(
            target=read_lidar_serial,
            args=(port, baudrate),
            daemon=True,
            name="lidar-serial-reader",
        )
        _reader_thread.start()
    print(f"[LIDAR] Thread started on {port}")
    return True


def stop_lidar():
    _stop_event.set()
    with _state_lock:
        serial_port = ser
    if serial_port is not None:
        try:
            serial_port.close()
        except Exception:
            pass
    print("[LIDAR] Stop requested")


def is_data_fresh(max_age_sec=LIDAR_DATA_TIMEOUT_SEC):
    with _state_lock:
        is_connected = connected
        last_point_time = last_valid_point_monotonic
    if not is_connected or last_point_time <= 0:
        return False
    return time.monotonic() - last_point_time <= max_age_sec


def get_status():
    with _state_lock:
        is_connected = connected
        last_point_time = last_valid_point_monotonic
    age = None
    if last_point_time > 0:
        age = max(0.0, time.monotonic() - last_point_time)
    return {
        "connected": is_connected,
        "data_fresh": is_data_fresh(),
        "last_point_age": age,
    }


def get_available_ports():
    return [port.device for port in serial.tools.list_ports.comports()]


if __name__ == "__main__":
    print("Available COM ports:", get_available_ports())
    selected_port = input("Enter COM port (e.g., COM3): ").strip()
    selected_baud = int(input("Enter baudrate (default 115200): ").strip() or "115200")
    start_lidar_thread(selected_port, selected_baud)

    try:
        while True:
            time.sleep(2)
            points = lidar_data.get_current_points()
            print(f"[DEBUG] Current points: {len(points)}, status={get_status()}")
            if points:
                print(f"[DEBUG] Sample point: {points[0]}")
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_lidar()
