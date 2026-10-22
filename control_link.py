"""Single-owner serial sender for the Raspberry Pi -> main ESP32 link."""

import os
import secrets
import threading
import time

import serial

from config import (
    CONTROL_HEARTBEAT_HZ,
    CONTROL_RECONNECT_DELAY_SEC,
    CONTROL_STARTUP_STOP_FRAMES,
    CONTROL_STOP_PULSE_SEC,
    CONTROL_WRITE_TIMEOUT_SEC,
    SERIAL_BAUD_CONTROL,
    SERIAL_PORT_CONTROL,
)
from control_protocol import (
    ControlCommand,
    ControlFlags,
    ControlMode,
    serialize_command,
)


class ControlLink:
    """The only Python component allowed to open/write the control port."""

    def __init__(self, shared_data, port=SERIAL_PORT_CONTROL):
        self._shared_data = shared_data
        self._port = port
        self._stop_event = threading.Event()
        self._thread = None
        self._serial = None
        self._state_lock = threading.Lock()
        self._connected = False
        self._session_id = secrets.randbits(32) or 1
        self._sequence = 0
        self._force_stop_until = 0.0
        self._identified = False
        self._diagnostic_buffer = ""
        self._last_error_text = ""
        self._last_error_log_monotonic = 0.0

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="esp32-control-sender",
        )
        self._thread.start()

    def request_stop(self, duration=CONTROL_STOP_PULSE_SEC):
        with self._state_lock:
            self._force_stop_until = max(
                self._force_stop_until,
                time.monotonic() + max(0.0, duration),
            )

    def stop(self):
        self.request_stop()
        self._stop_event.set()
        self._close_serial()

    def is_connected(self):
        with self._state_lock:
            return self._connected

    def get_status(self):
        with self._state_lock:
            return {
                "port": self._port,
                "connected": self._connected,
                "identified": self._identified,
            }

    def _set_connected(self, value):
        with self._state_lock:
            self._connected = value
            if not value:
                self._identified = False

    def _log_serial_error(self, exc):
        """Avoid filling journald while a configured USB device is unplugged."""
        message = str(exc)
        now = time.monotonic()
        if (
            message != self._last_error_text
            or now - self._last_error_log_monotonic >= 300.0
        ):
            print(f"[CtrlLink] Serial error: {message}")
            self._last_error_text = message
            self._last_error_log_monotonic = now

    def _consume_diagnostics(self, data):
        self._diagnostic_buffer += data.decode(errors="ignore")
        while "\n" in self._diagnostic_buffer:
            line, self._diagnostic_buffer = self._diagnostic_buffer.split("\n", 1)
            line = line.strip()
            identified = (
                line.startswith("ESP32 controller: iBUS GPIO")
                or (
                    line.startswith("status=")
                    and " CH1=" in line
                    and " CH2=" in line
                    and " pi_frames=" in line
                    and " seedSpeed=" in line
                )
            )
            if identified:
                with self._state_lock:
                    self._identified = True

    def _open_serial(self):
        # Configure DTR/RTS before opening so reconnect does not intentionally
        # pulse the ESP32 auto-reset circuit.
        port = serial.Serial(port=None)
        port.port = self._port
        port.baudrate = SERIAL_BAUD_CONTROL
        port.timeout = 0
        port.write_timeout = CONTROL_WRITE_TIMEOUT_SEC
        port.dsrdtr = False
        port.rtscts = False
        port.dtr = False
        port.rts = False
        if os.name != "nt" and hasattr(port, "exclusive"):
            port.exclusive = True
        port.open()
        port.reset_input_buffer()
        port.reset_output_buffer()
        return port

    def _close_serial(self):
        serial_port = self._serial
        self._serial = None
        self._set_connected(False)
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass

    def _forced_stop_active(self):
        with self._state_lock:
            return time.monotonic() < self._force_stop_until

    def _build_command(self, startup_stop):
        snapshot = self._shared_data.get_control_snapshot()
        lidar_valid = bool(snapshot["lidar_valid"])
        obstacle = bool(snapshot["lidar_detected"])
        manual_allow = lidar_valid and not obstacle

        flags = ControlFlags.NONE
        if lidar_valid:
            flags |= ControlFlags.LIDAR_VALID
        if obstacle:
            flags |= ControlFlags.OBSTACLE
        if manual_allow:
            flags |= ControlFlags.MANUAL_ALLOW

        route_running = bool(snapshot["route_running"])
        mode = ControlMode.AUTO if route_running else ControlMode.MANUAL_GUARD

        # AUTO motion/turn is deliberately left disabled until the drive
        # behavior is approved and tested.  The ESP32 also enforces AUTO_STOP.
        throttle = 0
        turn = 0
        must_stop = (
            startup_stop
            or self._forced_stop_active()
            or not lidar_valid
            or obstacle
            or route_running
        )
        if must_stop:
            flags |= ControlFlags.STOP_REQUEST
        if startup_stop:
            mode = ControlMode.STOP

        command = ControlCommand(
            session_id=self._session_id,
            sequence=self._sequence,
            mode=mode,
            flags=flags,
            throttle=throttle,
            turn=turn,
        )
        self._sequence = (self._sequence + 1) & 0xFFFF
        return command

    def _run(self):
        interval = 1.0 / CONTROL_HEARTBEAT_HZ

        while not self._stop_event.is_set():
            try:
                self._serial = self._open_serial()
                self._set_connected(True)
                startup_frames = CONTROL_STARTUP_STOP_FRAMES
                next_send = time.monotonic()
                print(
                    f"[CtrlLink] Connected to {self._port} "
                    f"@ {SERIAL_BAUD_CONTROL}; session={self._session_id}"
                )

                while not self._stop_event.is_set():
                    now = time.monotonic()
                    if now < next_send:
                        self._stop_event.wait(next_send - now)
                        continue

                    startup_stop = startup_frames > 0
                    packet = serialize_command(self._build_command(startup_stop))
                    self._serial.write(packet)
                    if startup_frames > 0:
                        startup_frames -= 1

                    # Drain current ESP32 text diagnostics. A future STATUS
                    # packet reader can replace this without adding a writer.
                    waiting = self._serial.in_waiting
                    if waiting:
                        self._consume_diagnostics(self._serial.read(waiting))

                    next_send += interval
                    if next_send < now - interval:
                        next_send = now + interval

            except Exception as exc:
                if not self._stop_event.is_set():
                    self._log_serial_error(exc)
                    self._shared_data.invalidate_motion_for_link_loss()
            finally:
                self._close_serial()

            if not self._stop_event.is_set():
                self._stop_event.wait(CONTROL_RECONNECT_DELAY_SEC)


__all__ = ["ControlLink"]
