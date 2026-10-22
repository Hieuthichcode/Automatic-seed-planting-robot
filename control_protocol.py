"""Protocol v1 for the Raspberry Pi -> main ESP32 control link.

Frames are COBS encoded and terminated by a zero byte.  The decoded command
frame is fixed at 20 bytes, including a CRC-16/CCITT-FALSE checksum.
"""

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import struct


MAGIC = b"RB"
PROTOCOL_VERSION = 1
MESSAGE_TYPE_COMMAND = 1
DECODED_COMMAND_LENGTH = 20

_COMMAND_WITHOUT_CRC = struct.Struct("<2sBBBBBBIHhh")
_CRC = struct.Struct("<H")


class ControlMode(IntEnum):
    STOP = 0
    MANUAL_DIRECT = 1
    MANUAL_GUARD = 2
    AUTO = 3


class ControlFlags(IntFlag):
    NONE = 0
    STOP_REQUEST = 1 << 0
    OBSTACLE = 1 << 1
    MANUAL_ALLOW = 1 << 2
    LIDAR_VALID = 1 << 3


KNOWN_FLAG_MASK = int(
    ControlFlags.STOP_REQUEST
    | ControlFlags.OBSTACLE
    | ControlFlags.MANUAL_ALLOW
    | ControlFlags.LIDAR_VALID
)


@dataclass(frozen=True)
class ControlCommand:
    session_id: int
    sequence: int
    mode: ControlMode
    flags: ControlFlags
    throttle: int = 0
    turn: int = 0


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def cobs_encode(data: bytes) -> bytes:
    output = bytearray([0])
    code_index = 0
    code = 1

    for value in data:
        if value == 0:
            output[code_index] = code
            code_index = len(output)
            output.append(0)
            code = 1
        else:
            output.append(value)
            code += 1
            if code == 0xFF:
                output[code_index] = code
                code_index = len(output)
                output.append(0)
                code = 1

    output[code_index] = code
    return bytes(output)


def cobs_decode(data: bytes) -> bytes:
    if not data:
        raise ValueError("empty COBS frame")

    output = bytearray()
    index = 0
    while index < len(data):
        code = data[index]
        if code == 0:
            raise ValueError("zero byte inside COBS frame")
        index += 1

        end = index + code - 1
        if end > len(data):
            raise ValueError("truncated COBS block")
        output.extend(data[index:end])
        index = end

        if code != 0xFF and index < len(data):
            output.append(0)

    return bytes(output)


def serialize_command(command: ControlCommand) -> bytes:
    if not 0 <= command.session_id <= 0xFFFFFFFF:
        raise ValueError("session_id outside uint32 range")
    if not 0 <= command.sequence <= 0xFFFF:
        raise ValueError("sequence outside uint16 range")
    if not -1000 <= command.throttle <= 1000:
        raise ValueError("throttle outside -1000..1000")
    if not -1000 <= command.turn <= 1000:
        raise ValueError("turn outside -1000..1000")
    if int(command.flags) & ~KNOWN_FLAG_MASK:
        raise ValueError("unknown control flag")

    payload = _COMMAND_WITHOUT_CRC.pack(
        MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_TYPE_COMMAND,
        DECODED_COMMAND_LENGTH,
        int(command.mode),
        int(command.flags),
        0,
        command.session_id,
        command.sequence,
        command.throttle,
        command.turn,
    )
    decoded = payload + _CRC.pack(crc16_ccitt_false(payload))
    return cobs_encode(decoded) + b"\x00"


def deserialize_command(packet: bytes) -> ControlCommand:
    encoded = packet[:-1] if packet.endswith(b"\x00") else packet
    decoded = cobs_decode(encoded)
    if len(decoded) != DECODED_COMMAND_LENGTH:
        raise ValueError("invalid command length")

    payload = decoded[:-2]
    received_crc = _CRC.unpack(decoded[-2:])[0]
    if crc16_ccitt_false(payload) != received_crc:
        raise ValueError("CRC mismatch")

    (
        magic,
        version,
        message_type,
        frame_length,
        mode_value,
        flags_value,
        reserved,
        session_id,
        sequence,
        throttle,
        turn,
    ) = _COMMAND_WITHOUT_CRC.unpack(payload)

    if magic != MAGIC:
        raise ValueError("invalid magic")
    if version != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol version")
    if message_type != MESSAGE_TYPE_COMMAND:
        raise ValueError("unsupported message type")
    if frame_length != DECODED_COMMAND_LENGTH or reserved != 0:
        raise ValueError("invalid frame metadata")
    if flags_value & ~KNOWN_FLAG_MASK:
        raise ValueError("unknown control flag")
    if not -1000 <= throttle <= 1000 or not -1000 <= turn <= 1000:
        raise ValueError("control value outside valid range")

    try:
        mode = ControlMode(mode_value)
    except ValueError as exc:
        raise ValueError("unsupported control mode") from exc

    return ControlCommand(
        session_id=session_id,
        sequence=sequence,
        mode=mode,
        flags=ControlFlags(flags_value),
        throttle=throttle,
        turn=turn,
    )
