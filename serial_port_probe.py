"""Safe serial-port discovery and device identification helpers."""

import secrets
import time

import serial
from serial.tools import list_ports

from Read_Serial import parse_gps_imu_line
from Read_lidar import parse_lidar_line
from control_protocol import (
    ControlCommand,
    ControlFlags,
    ControlMode,
    serialize_command,
)


DEVICE_LABELS = {
    "gps_imu": "GPS/IMU",
    "controller": "ESP32 điều khiển",
    "lidar": "Lidar",
}

# The main motor-controller board in this project uses a CH340 USB-UART
# adapter. Unlike the ESP32-C3 sensor boards, its current firmware does not
# always emit diagnostics, so this is the non-invasive identification fallback.
CONTROLLER_USB_IDS = {(0x1A86, 0x7523)}


def available_serial_ports():
    """Return stable metadata used to populate the configuration dropdowns."""
    ports = []
    for item in sorted(list_ports.comports(), key=lambda value: value.device):
        ports.append({
            "device": item.device,
            "description": item.description or "Cổng serial",
            "hwid": item.hwid or "",
            "vid": item.vid,
            "pid": item.pid,
            "serial_number": item.serial_number or "",
        })
    return ports


def port_matches_device_usb(device_type, port):
    """Use a known USB adapter identity when firmware cannot answer safely."""
    if device_type != "controller":
        return False

    wanted_port = port.upper()
    for item in list_ports.comports():
        if item.device.upper() == wanted_port:
            return (item.vid, item.pid) in CONTROLLER_USB_IDS
    return False


def line_matches_device(device_type, line):
    """Recognize a device using only its firmware output format."""
    text = line.strip()
    if not text:
        return False
    if device_type == "gps_imu":
        return (
            parse_gps_imu_line(text) is not None
            or text.startswith("BOOT,ESP32-C3,MPU")
        )
    if device_type == "lidar":
        return (
            parse_lidar_line(text) is not None
            or text.startswith("LiDAR model ")
        )
    if device_type == "controller":
        return (
            text.startswith("ESP32 controller: iBUS GPIO")
            or (
                text.startswith("status=")
                and " CH1=" in text
                and " CH2=" in text
                and " pi_frames=" in text
                and " seedSpeed=" in text
            )
        )
    return False


def _open_without_reset(port, baudrate):
    serial_port = serial.Serial(port=None)
    serial_port.port = port
    serial_port.baudrate = baudrate
    serial_port.timeout = 0.15
    serial_port.write_timeout = 0.2
    serial_port.dsrdtr = False
    serial_port.rtscts = False
    serial_port.dtr = False
    serial_port.rts = False
    serial_port.open()
    serial_port.reset_input_buffer()
    return serial_port


def _safe_stop_packet(session_id, sequence):
    return serialize_command(ControlCommand(
        session_id=session_id,
        sequence=sequence,
        mode=ControlMode.STOP,
        flags=ControlFlags.STOP_REQUEST,
        throttle=0,
        turn=0,
    ))


def probe_serial_port(device_type, port, baudrate=115200, timeout_sec=3.0):
    """Open an unused port and verify that its data matches the selected role.

    The controller probe only transmits STOP packets, so probing cannot request
    movement even if it is run while the motor power stage is connected.
    """
    if device_type not in DEVICE_LABELS:
        raise ValueError("Loại thiết bị không hợp lệ")
    if not port or not isinstance(port, str):
        raise ValueError("Cổng COM không hợp lệ")

    if device_type == "controller" and port_matches_device_usb(device_type, port):
        return {
            "success": True,
            "message": "Đã nhận đúng bộ điều khiển qua USB CH340",
            "sample": "USB VID:PID=1A86:7523 (CH340)",
        }

    serial_port = None
    samples = []
    try:
        serial_port = _open_without_reset(port, baudrate)
        deadline = time.monotonic() + timeout_sec
        session_id = secrets.randbits(32) or 1
        sequence = 0
        next_stop_send = 0.0

        while time.monotonic() < deadline:
            now = time.monotonic()
            if device_type == "controller" and now >= next_stop_send:
                serial_port.write(_safe_stop_packet(session_id, sequence))
                sequence = (sequence + 1) & 0xFFFF
                next_stop_send = now + 0.04

            raw = serial_port.readline()
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            if line and len(samples) < 3:
                samples.append(line[:160])
            if line_matches_device(device_type, line):
                return {
                    "success": True,
                    "message": f"Đã nhận đúng dữ liệu {DEVICE_LABELS[device_type]}",
                    "sample": line[:160],
                }

        return {
            "success": False,
            "message": (
                f"Không nhận được dữ liệu đúng định dạng "
                f"{DEVICE_LABELS[device_type]} trong {timeout_sec:.0f} giây"
            ),
            "sample": samples[0] if samples else "",
        }
    except (OSError, serial.SerialException) as exc:
        return {
            "success": False,
            "message": f"Không thể mở {port}: {exc}",
            "sample": "",
        }
    finally:
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass


__all__ = [
    "DEVICE_LABELS",
    "available_serial_ports",
    "line_matches_device",
    "port_matches_device_usb",
    "probe_serial_port",
]
