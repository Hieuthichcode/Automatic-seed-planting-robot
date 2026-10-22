"""Production entry point for the Raspberry Pi systemd service."""

from waitress import serve

from app import (
    ACTIVE_SERIAL_PORTS,
    FLASK_HOST,
    FLASK_PORT,
    LIDAR_BAUDRATE,
    MdnsAdvertiser,
    Read_lidar,
    app,
    arm_control,
    control_link,
)


def main():
    print("[SERVICE] Starting Raspberry Pi robot services")
    Read_lidar.start_lidar_thread(
        ACTIVE_SERIAL_PORTS["lidar"],
        LIDAR_BAUDRATE,
    )
    arm_control.init_servos()
    mdns = MdnsAdvertiser(FLASK_PORT)
    try:
        mdns.start()
        serve(
            app,
            host=FLASK_HOST,
            port=FLASK_PORT,
            threads=6,
            channel_timeout=30,
            cleanup_interval=10,
        )
    finally:
        print("[SERVICE] Stopping Raspberry Pi robot services")
        control_link.stop()
        Read_lidar.stop_lidar()
        arm_control.cleanup()
        mdns.stop()


if __name__ == "__main__":
    main()
