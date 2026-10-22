"""Advertise the robot web interface as control.local on the local network."""

import ipaddress
import socket

from zeroconf import IPVersion, ServiceInfo, Zeroconf


MDNS_HOSTNAME = "control.local."
MDNS_SERVICE_TYPE = "_http._tcp.local."
MDNS_SERVICE_NAME = "Robot Control._http._tcp.local."


def local_ipv4_addresses():
    """Return usable local IPv4 addresses without advertising loopback."""
    addresses = set()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            addresses.add(probe.getsockname()[0])
    except OSError:
        pass

    try:
        for address in socket.gethostbyname_ex(socket.gethostname())[2]:
            addresses.add(address)
    except OSError:
        pass

    return sorted(
        address
        for address in addresses
        if not ipaddress.ip_address(address).is_loopback
        and not ipaddress.ip_address(address).is_link_local
    )


class MdnsAdvertiser:
    def __init__(self, port):
        self.port = int(port)
        self._zeroconf = None
        self._service_info = None

    def start(self):
        addresses = local_ipv4_addresses()
        if not addresses:
            print("[mDNS] Không tìm thấy địa chỉ IPv4 LAN; bỏ qua control.local")
            return False

        self._zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
        self._service_info = ServiceInfo(
            MDNS_SERVICE_TYPE,
            MDNS_SERVICE_NAME,
            addresses=[socket.inet_aton(address) for address in addresses],
            port=self.port,
            properties={"path": "/"},
            server=MDNS_HOSTNAME,
        )
        self._zeroconf.register_service(
            self._service_info,
            allow_name_change=True,
        )
        port_suffix = "" if self.port == 80 else f":{self.port}"
        print(
            f"[mDNS] Truy cập tại http://control.local{port_suffix} "
            f"({', '.join(addresses)})"
        )
        return True

    def stop(self):
        if self._zeroconf is None:
            return
        try:
            if self._service_info is not None:
                self._zeroconf.unregister_service(self._service_info)
        finally:
            self._zeroconf.close()
            self._zeroconf = None
            self._service_info = None


__all__ = ["MdnsAdvertiser", "local_ipv4_addresses"]
