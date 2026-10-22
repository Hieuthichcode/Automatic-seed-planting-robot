import pathlib
import sys
import unittest


MODULE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from control_link import ControlLink  # noqa: E402
from control_protocol import ControlFlags, ControlMode  # noqa: E402


class FakeSharedData:
    def __init__(self, **overrides):
        self.snapshot = {
            "route_running": False,
            "move_cmd": 0,
            "lidar_detected": False,
            "lidar_min_distance": 9999,
            "lidar_valid": True,
        }
        self.snapshot.update(overrides)
        self.invalidated = False

    def get_control_snapshot(self):
        return dict(self.snapshot)

    def invalidate_motion_for_link_loss(self):
        self.invalidated = True


class ControlLinkStateTests(unittest.TestCase):
    def test_clear_lidar_allows_guarded_manual(self):
        command = ControlLink(FakeSharedData())._build_command(False)
        self.assertEqual(command.mode, ControlMode.MANUAL_GUARD)
        self.assertTrue(command.flags & ControlFlags.LIDAR_VALID)
        self.assertTrue(command.flags & ControlFlags.MANUAL_ALLOW)
        self.assertFalse(command.flags & ControlFlags.STOP_REQUEST)

    def test_obstacle_denies_guarded_manual(self):
        command = ControlLink(FakeSharedData(
            lidar_detected=True,
            lidar_min_distance=200,
        ))._build_command(False)
        self.assertTrue(command.flags & ControlFlags.OBSTACLE)
        self.assertTrue(command.flags & ControlFlags.STOP_REQUEST)
        self.assertFalse(command.flags & ControlFlags.MANUAL_ALLOW)

    def test_invalid_lidar_fails_safe(self):
        command = ControlLink(FakeSharedData(
            lidar_valid=False,
        ))._build_command(False)
        self.assertTrue(command.flags & ControlFlags.STOP_REQUEST)
        self.assertFalse(command.flags & ControlFlags.LIDAR_VALID)

    def test_auto_motion_remains_todo_and_stopped(self):
        command = ControlLink(FakeSharedData(
            route_running=True,
            move_cmd=1,
        ))._build_command(False)
        self.assertEqual(command.mode, ControlMode.AUTO)
        self.assertEqual(command.throttle, 0)
        self.assertEqual(command.turn, 0)
        self.assertTrue(command.flags & ControlFlags.STOP_REQUEST)

    def test_startup_frames_are_stop_mode(self):
        command = ControlLink(FakeSharedData())._build_command(True)
        self.assertEqual(command.mode, ControlMode.STOP)
        self.assertTrue(command.flags & ControlFlags.STOP_REQUEST)


if __name__ == "__main__":
    unittest.main()
