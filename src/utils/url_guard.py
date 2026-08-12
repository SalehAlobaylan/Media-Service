"""SSRF guard for server-side URL fetches.

The transcription and image-embedding endpoints download caller/pipeline-supplied
URLs server-side. Without validation an attacker who can steer a URL here could
reach internal hosts (e.g. http://169.254.169.254/… cloud metadata) or other
services inside the trust boundary. This module enforces an http(s)-only scheme
and rejects any URL whose host resolves into a private, loopback, link-local or
otherwise non-public address range.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_ALLOWED_SCHEMES = {"http", "https"}


class UnsafeURLError(ValueError):
    """Raised when a URL is not safe to fetch server-side."""


def _is_public_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def validate_public_url(url: str) -> str:
    """Return ``url`` unchanged if safe to fetch, else raise UnsafeURLError.

    Enforces an http(s) scheme and verifies that every address the hostname
    resolves to is publicly routable, blocking SSRF to internal services and
    cloud metadata endpoints.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeURLError(
            f"URL scheme '{parsed.scheme}' is not allowed (only http/https)"
        )

    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL has no host")

    if parsed.username or parsed.password:
        raise UnsafeURLError("URL userinfo is not allowed")
    validate_public_host(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    return url


def validate_public_host(host: str, port: int) -> str:
    """Resolve and select a public address for the imminent TCP connection."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"could not resolve host '{host}'") from exc

    resolved = sorted({info[4][0] for info in infos})
    if not resolved:
        raise UnsafeURLError(f"could not resolve host '{host}'")

    for ip in resolved:
        if not _is_public_ip(ip):
            raise UnsafeURLError(f"host '{host}' resolves to non-public address {ip}")

    return resolved[0]


def safe_url_host(url: str) -> str:
    """A log-safe source label that never includes paths, userinfo, or queries."""
    return urlparse(url).hostname or "invalid-host"
