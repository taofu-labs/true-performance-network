"""Validation for operator-supplied base URLs (leader validator, coordinator)."""
from urllib.parse import urlparse


class InvalidBaseUrl(ValueError):
    """Raised when a base URL is not a usable http(s) origin."""


def validate_base_url(url: str, *, setting_name: str = "base URL") -> str:
    """Return `url` stripped of trailing slashes, or raise InvalidBaseUrl."""
    if not url or not url.strip():
        raise InvalidBaseUrl(f"{setting_name} is empty")

    cleaned = url.strip()
    parsed = urlparse(cleaned)

    if parsed.scheme not in ("http", "https"):
        raise InvalidBaseUrl(
            f"{setting_name} must start with http:// or https:// — got {cleaned!r}"
        )

    if not parsed.netloc:
        raise InvalidBaseUrl(f"{setting_name} has no host — got {cleaned!r}")

    host = parsed.netloc
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    if host.count(":") > 1:
        raise InvalidBaseUrl(
            f"{setting_name} has a malformed host {parsed.netloc!r} "
            f"(doubled scheme?) — got {cleaned!r}"
        )
    if ":" in host:
        hostname, _, port = host.partition(":")
        if not hostname:
            raise InvalidBaseUrl(f"{setting_name} has no host — got {cleaned!r}")
        if not port:
            raise InvalidBaseUrl(
                f"{setting_name} has a malformed host {parsed.netloc!r} "
                f"(doubled scheme?) — got {cleaned!r}"
            )
        if not port.isdigit():
            raise InvalidBaseUrl(
                f"{setting_name} has a non-numeric port {port!r} — got {cleaned!r}"
            )

    if parsed.path.strip("/"):
        if "//" in parsed.path:
            raise InvalidBaseUrl(
                f"{setting_name} contains an embedded scheme in its path "
                f"— got {cleaned!r}"
            )

    return cleaned.rstrip("/")
