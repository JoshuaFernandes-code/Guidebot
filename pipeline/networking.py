import ipaddress
import socket


def discover_stream_lan_host(configured_host: str = "") -> str | None:
    if configured_host:
        try:
            address = ipaddress.ip_address(configured_host)
        except ValueError:
            print(f"[Stream] Invalid STREAM_LAN_HOST IPv4 address: {configured_host!r}")
            return None
        if address.version != 4:
            print("[Stream] STREAM_LAN_HOST must be an IPv4 address.")
            return None
        return str(address)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("1.1.1.1", 80))
            address = ipaddress.ip_address(probe.getsockname()[0])
        return str(address) if address.version == 4 and not address.is_unspecified else None
    except OSError as exc:
        print(f"[Stream] Could not detect LAN stream address; using tunnel for video: {exc}")
        return None
