import pathlib
import sys
import unittest


MODULE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from serial_port_probe import (  # noqa: E402
    line_matches_device,
    port_matches_device_usb,
)


class SerialPortProbeTests(unittest.TestCase):
    def test_gps_imu_signature_is_not_confused_with_other_devices(self):
        line = (
            "12345,-12.50,1.00,2.00,0.1,0.2,1.0,0.0,0.0,0.1,30.0,1,"
            "20.962059,105.748392,15.0,3.5,90.0,10,0.9,100"
        )
        self.assertTrue(line_matches_device("gps_imu", line))
        self.assertFalse(line_matches_device("lidar", line))
        self.assertFalse(line_matches_device("controller", line))

    def test_lidar_signature_is_not_confused_with_other_devices(self):
        line = "450.5 90.0 123"
        self.assertTrue(line_matches_device("lidar", line))
        self.assertFalse(line_matches_device("gps_imu", line))
        self.assertFalse(line_matches_device("controller", line))

    def test_controller_signature_is_not_confused_with_other_devices(self):
        line = (
            "status=RC_STOP frames=20 crc_errors=0 age_ms=5 CH1=1500 CH2=1500 "
            "CH9=1000 CH10=1000 throttle=0 turn=0 leftPWM=0 rightPWM=0 "
            "pi_frames=12 pi_invalid=0 pi_duplicate=0 pi_age_ms=10 pump1=0 "
            "pump2=0 seed=0 seedSpeed=0"
        )
        self.assertTrue(line_matches_device("controller", line))
        self.assertFalse(line_matches_device("gps_imu", line))
        self.assertFalse(line_matches_device("lidar", line))

    def test_controller_usb_identity_uses_ch340_vid_pid(self):
        from unittest.mock import patch

        class Port:
            device = "COM48"
            vid = 0x1A86
            pid = 0x7523

        with patch("serial_port_probe.list_ports.comports", return_value=[Port()]):
            self.assertTrue(port_matches_device_usb("controller", "com48"))
            self.assertFalse(port_matches_device_usb("gps_imu", "COM48"))
            self.assertFalse(port_matches_device_usb("controller", "COM43"))


if __name__ == "__main__":
    unittest.main()
