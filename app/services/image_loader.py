"""Async image fetch with SSRF guard, size cap, and EXIF rotation.

Threat model:

- A user pastes an arbitrary URL in the request body. We must not let
  that URL be used to probe internal services (SSRF) or fill memory
  with a 5 GB GIF (DoS).
- The image bytes themselves are decoded by Pillow, which has a
  reasonable history with malformed inputs. We additionally cap by
  Content-Length and stream size before passing bytes to Pillow.

Failures map to a small hierarchy of typed exceptions
(:class:`ImageLoaderError` and subclasses) that the FastAPI app
converts to specific HTTP 400 responses with stable ``error_code`` tags.
"""

from __future__ import annotations

import io
import ipaddress
import socket
from typing import Tuple

import httpx
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import HttpUrl


# --- Exception hierarchy -------------------------------------------------


class ImageLoaderError(Exception):
    """Base class. Anything raised from this module is HTTP 400."""


class URLBlockedError(ImageLoaderError):
    """SSRF guard rejected the URL (private/loopback/link-local IP, etc.)."""


class URLUnreachableError(ImageLoaderError):
    """DNS resolution failed, connection refused, or response wasn't 2xx."""


class InvalidContentTypeError(ImageLoaderError):
    """Response Content-Type does not start with ``image/``."""


class ImageTooLargeError(ImageLoaderError):
    """Image exceeded the configured size cap."""


class ImageDecodeError(ImageLoaderError):
    """Bytes were downloaded but Pillow could not decode them."""


# --- SSRF guard ----------------------------------------------------------


def _is_blocked_ip(addr: str) -> bool:
    """Return True if ``addr`` is private, loopback, link-local, or reserved.

    Covers (IPv4 and IPv6):
      - 127/8 and ::1            (loopback)
      - 10/8, 172.16/12, 192.168/16, fc00::/7  (private)
      - 169.254/16, fe80::/10    (link-local)
      - multicast / reserved / unspecified

    Args:
        addr: A numeric IP literal (NOT a hostname). Caller resolves first.

    Returns:
        True if the address should be refused.
    """
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        # Anything we can't parse, refuse (defense-in-depth).
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _check_url_safe(url: HttpUrl, *, block_private_ips: bool) -> Tuple[str, str]:
    """Validate scheme and host, then resolve and SSRF-check.

    Pydantic's ``HttpUrl`` already enforces http/https and a non-empty
    host, but we re-check explicitly so the error path is local to this
    module and not at the schema layer.

    Args:
        url: User-supplied URL.
        block_private_ips: Toggle from settings. When False, only
            literal "localhost" and obviously-bogus hosts are rejected.

    Returns:
        ``(scheme, host)`` as plain strings.

    Raises:
        URLBlockedError: SSRF guard tripped.
        URLUnreachableError: DNS failed.
    """
    # Pydantic v2 HttpUrl exposes its parts as attributes on the URL object.
    scheme = url.scheme
    host = url.host
    if scheme not in ("http", "https"):
        raise URLBlockedError(f"unsupported URL scheme: {scheme!r}")
    if not host:
        raise URLBlockedError("URL has no host")

    # Block obvious local hostnames before we even try to resolve.
    host_lc = host.lower()
    if host_lc in ("localhost", "ip6-localhost", "ip6-loopback") or host_lc.endswith(
        (".localhost", ".local")
    ):
        raise URLBlockedError(f"refusing to fetch from local hostname: {host!r}")

    if not block_private_ips:
        return scheme, host

    # Resolve all addresses for the host (handles round-robin DNS, IPv6).
    try:
        addrinfo = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise URLUnreachableError(f"DNS resolution failed for {host!r}: {e}") from e

    for family, _stype, _proto, _canon, sockaddr in addrinfo:
        addr = sockaddr[0]
        if _is_blocked_ip(addr):
            raise URLBlockedError(
                f"refusing to fetch from non-public IP {addr!r} (resolved from {host!r})"
            )

    return scheme, host


# --- Public entry point --------------------------------------------------


async def fetch_image(
    url: HttpUrl,
    *,
    timeout_s: float,
    max_size_bytes: int,
    block_private_ips: bool = True,
) -> NDArray[np.uint8]:
    """Download, decode, and EXIF-rotate the image at ``url``.

    Args:
        url: Validated HTTP/HTTPS URL.
        timeout_s: Total request timeout (connect + read + write).
        max_size_bytes: Hard cap on download size. Enforced both via
            ``Content-Length`` (cheap reject) and while streaming
            (the header can lie or be absent).
        block_private_ips: If True (default), refuse private/loopback/
            link-local addresses. Disable only in trusted environments.

    Returns:
        A ``(h, w, 3)`` uint8 RGB numpy array, EXIF-rotated.

    Raises:
        URLBlockedError: SSRF guard blocked the URL.
        URLUnreachableError: DNS / connection / non-2xx response.
        InvalidContentTypeError: Response is not ``image/*``.
        ImageTooLargeError: Image exceeded ``max_size_bytes``.
        ImageDecodeError: Bytes downloaded but not decodable.
    """
    _check_url_safe(url, block_private_ips=block_private_ips)

    timeout = httpx.Timeout(timeout_s)
    # Don't follow redirects to URLs that bypass our SSRF check on the
    # original host. We accept up to 3 redirects only if every hop
    # resolves to a public IP — re-check explicitly. For simplicity we
    # disable redirects entirely; if you need them, add per-hop checks.
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            async with client.stream("GET", str(url)) as response:
                if response.status_code >= 400:
                    raise URLUnreachableError(
                        f"upstream returned HTTP {response.status_code}"
                    )
                if response.status_code in (301, 302, 303, 307, 308):
                    # Follow-redirects is off, so a redirect means refuse.
                    raise URLUnreachableError(
                        f"upstream redirected (HTTP {response.status_code}); "
                        "redirects are disabled for SSRF safety"
                    )

                ctype = response.headers.get("content-type", "").lower()
                if not ctype.startswith("image/"):
                    raise InvalidContentTypeError(
                        f"expected image/*, got Content-Type: {ctype or 'missing'!r}"
                    )

                # Cheap up-front reject if the server told us the size.
                cl = response.headers.get("content-length")
                if cl is not None:
                    try:
                        if int(cl) > max_size_bytes:
                            raise ImageTooLargeError(
                                f"image is {cl} bytes, exceeds cap of {max_size_bytes}"
                            )
                    except ValueError:
                        # Malformed Content-Length; fall through to stream check.
                        pass

                # Stream and enforce the cap as we go.
                buf = bytearray()
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_size_bytes:
                        raise ImageTooLargeError(
                            f"image exceeded cap of {max_size_bytes} bytes mid-stream"
                        )
                    buf.extend(chunk)
                body = bytes(buf)
    except httpx.ConnectError as e:
        raise URLUnreachableError(f"connection failed: {e}") from e
    except httpx.TimeoutException as e:
        raise URLUnreachableError(f"request timed out after {timeout_s}s: {e}") from e
    except httpx.RequestError as e:
        # Catch-all for non-connect/timeout request issues (proxies, ssl).
        raise URLUnreachableError(f"HTTP request failed: {e}") from e

    # Decode + EXIF transpose
    try:
        with Image.open(io.BytesIO(body)) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            img_rgb = np.array(pil_img.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ImageDecodeError(f"could not decode image: {e}") from e

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        # Should be impossible after .convert('RGB'), but defensive.
        raise ImageDecodeError(f"unexpected image shape: {img_rgb.shape}")

    return img_rgb
