import pathlib
import sys
import unittest


MODULE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from control_protocol import (  # noqa: E402
    ControlCommand,
    ControlFlags,
    ControlMode,
    cobs_decode,
    cobs_encode,
    deserialize_command,
    serialize_command,
)


class ControlProtocolTests(unittest.TestCase):
    def test_round_trip(self):
        command = ControlCommand(
            session_id=0x12345678,
            sequence=65535,
            mode=ControlMode.MANUAL_GUARD,
            flags=(
                ControlFlags.MANUAL_ALLOW
                | ControlFlags.LIDAR_VALID
            ),
            throttle=-1000,
            turn=1000,
        )
        self.assertEqual(deserialize_command(serialize_command(command)), command)

    def test_known_cross_language_vector(self):
        command = ControlCommand(
            session_id=0x12345678,
            sequence=0x0102,
            mode=ControlMode.MANUAL_GUARD,
            flags=ControlFlags.MANUAL_ALLOW | ControlFlags.LIDAR_VALID,
            throttle=-1000,
            turn=1000,
        )
        self.assertEqual(
            serialize_command(command).hex(),
            "085242010114020c0d78563412020118fce8030a1900",
        )

    def test_cobs_round_trip_with_zero_bytes(self):
        payload = bytes([0, 1, 0, 2, 3, 0, 0, 255])
        self.assertEqual(cobs_decode(cobs_encode(payload)), payload)

    def test_crc_error_is_rejected(self):
        packet = bytearray(serialize_command(ControlCommand(
            session_id=7,
            sequence=8,
            mode=ControlMode.STOP,
            flags=ControlFlags.STOP_REQUEST,
        )))
        packet[5] ^= 0x20
        with self.assertRaises(ValueError):
            deserialize_command(bytes(packet))

    def test_truncated_packet_is_rejected(self):
        packet = serialize_command(ControlCommand(
            session_id=1,
            sequence=1,
            mode=ControlMode.STOP,
            flags=ControlFlags.STOP_REQUEST,
        ))
        with self.assertRaises(ValueError):
            deserialize_command(packet[:-3] + b"\x00")

    def test_out_of_range_command_is_rejected_before_serializing(self):
        with self.assertRaises(ValueError):
            serialize_command(ControlCommand(
                session_id=1,
                sequence=1,
                mode=ControlMode.AUTO,
                flags=ControlFlags.NONE,
                throttle=1001,
            ))


if __name__ == "__main__":
    unittest.main()
