import math
import pathlib
import sys
import unittest


MODULE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from Read_Serial import parse_gps_imu_line, signed_angle_delta  # noqa: E402
from Read_lidar import parse_lidar_line  # noqa: E402


class SensorParserTests(unittest.TestCase):
    def test_current_gps_imu_csv(self):
        line = (
            "12345,-12.50,1.00,2.00,0.1,0.2,1.0,0.0,0.0,0.1,30.0,1,"
            "20.962059,105.748392,15.0,3.5,90.0,10,0.9,100"
        )
        parsed = parse_gps_imu_line(line)
        self.assertEqual(parsed["source_ms"], 12345)
        self.assertAlmostEqual(parsed["yaw"], -12.5)
        self.assertAlmostEqual(parsed["lat"], 20.962059)
        self.assertAlmostEqual(parsed["lon"], 105.748392)
        self.assertTrue(parsed["gps_valid"])

    def test_no_gps_fix_keeps_yaw_but_not_nan_location(self):
        line = (
            "100,179.0,0,0,0,0,1,0,0,0,30,0,nan,nan,nan,nan,nan,nan,nan,-1"
        )
        parsed = parse_gps_imu_line(line)
        self.assertEqual(parsed["yaw"], 179.0)
        self.assertFalse(parsed["gps_valid"])
        self.assertNotIn("lat", parsed)
        self.assertNotIn("lon", parsed)

    def test_gps_header_is_ignored(self):
        self.assertIsNone(parse_gps_imu_line(
            "ms,yaw_deg,pitch_deg,roll_deg,ax,ay,az,gx,gy,gz,temp,fix,lat,lon,alt,speed,course,sat,hdop,age"
        ))

    def test_yaw_delta_wraps_at_180(self):
        self.assertAlmostEqual(signed_angle_delta(-179.0, 179.0), 2.0)
        self.assertAlmostEqual(signed_angle_delta(179.0, -179.0), -2.0)

    def test_lidar_three_field_firmware_format(self):
        self.assertEqual(parse_lidar_line("450.5 90.0 123"), (450.5, 90.0, 123.0))

    def test_lidar_two_field_compatibility(self):
        self.assertEqual(parse_lidar_line("500 -10"), (500.0, 350.0, 0.0))

    def test_lidar_status_text_and_nonfinite_values_are_ignored(self):
        self.assertIsNone(parse_lidar_line("Scan completed; scans-per-second 5"))
        self.assertIsNone(parse_lidar_line("nan 10 20"))
        self.assertIsNone(parse_lidar_line("0 10 20"))


if __name__ == "__main__":
    unittest.main()
