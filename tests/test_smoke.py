"""Smoke tests.

These tests run without network access and without instantiating a real
MediaPipe ``FaceLandmarker``. The ``app_client`` fixture stubs the
detector class so the FastAPI lifespan completes instantly even when
the ``.task`` bundle isn't available; HTTP-level tests for the image
loader use ``httpx.MockTransport`` so we can simulate slow / lying /
oversized servers without touching the network.

What we cover (the production-bite set):

- Every module imports (catches syntax errors, missing deps, circular
  imports, broken module-level asserts in landmarks.py).
- ``GET /health`` returns 200 ``{"status": "ok"}``.
- ``GET /`` returns the API banner.
- ``POST /analyze`` rejects malformed URLs (422, Pydantic).
- ``POST /analyze`` rejects unsupported schemes (422, Pydantic HttpUrl).
- ``POST /analyze`` rejects local hostnames (400, SSRF).
- ``POST /analyze`` rejects IPv4 private / loopback / link-local (400, SSRF).
- ``POST /analyze`` rejects IPv6 loopback (400, SSRF).
- ``POST /analyze`` returns distinct ``error_code`` for ``NoFaceDetectedError``
  vs ``MultipleFacesDetectedError`` so callers can branch on them.
- ``image_loader`` rejects non-``image/*`` Content-Type (400).
- ``image_loader`` rejects images over the size cap via the
  ``Content-Length`` cheap-reject path.
- ``image_loader`` rejects images over the size cap via the streaming
  path when the server LIES about ``Content-Length``.
- ``image_loader`` rejects undecodable image bytes (400).
- Per-zone severity bucketing matches the threshold tables.
- ``measure_zone`` handles an empty skeleton gracefully.

End-to-end tests with a real selfie are intentionally NOT in the smoke
suite — they require ~3.8 MB of model download and ~1 s of inference.
Add an integration-tier test directory if you need that coverage.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Iterator

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient


# --- Module-import sanity ------------------------------------------------


def test_imports_all_modules() -> None:
    """Importing each module shouldn't raise.

    Catches: syntax errors, missing dependencies, circular imports,
    and the ZONE_PRIORITY / ZONE_ITERATION_ORDER asserts in
    ``app.core.landmarks``.
    """
    import app  # noqa: F401
    import app.config  # noqa: F401
    import app.schemas  # noqa: F401
    import app.core  # noqa: F401
    import app.core.landmarks  # noqa: F401
    import app.core.masks  # noqa: F401
    import app.core.measure  # noqa: F401
    import app.core.preprocess  # noqa: F401
    import app.core.ridges  # noqa: F401
    import app.core.severity  # noqa: F401
    import app.services  # noqa: F401
    import app.services.image_loader  # noqa: F401
    import app.services.wrinkle_detector  # noqa: F401
    import app.main  # noqa: F401


def test_zone_lists_consistent() -> None:
    """Belt-and-braces: re-check the assertions in landmarks.py."""
    from app.core import landmarks as lm

    assert set(lm.ZONE_PRIORITY) == set(lm.ZONE_ITERATION_ORDER)
    assert len(lm.ZONE_PRIORITY) == 14


# --- HTTP smoke fixture --------------------------------------------------


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """TestClient backed by a stub detector. No network, no MediaPipe init."""
    # 1) Place a dummy `.task` file so the auto-download branch is skipped.
    model_path = tmp_path / "face_landmarker.task"
    model_path.write_bytes(b"stub")

    from app import main as main_mod
    from app.config import settings

    monkeypatch.setattr(settings, "model_path", model_path)

    # 2) Stub the detector class so MediaPipe is never invoked. The
    #    constructor accepts the same kwargs but does nothing. The
    #    `analyze` method is overridden per-test where needed.
    class _StubDetector:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def close(self) -> None:
            pass

        def analyze(self, _img_rgb):
            raise NotImplementedError("smoke tests do not invoke the pipeline")

    # main.py imported `WrinkleDetector` by name into its module namespace,
    # so we patch that binding (the lifespan calls `WrinkleDetector(...)`
    # via that local lookup).
    monkeypatch.setattr(main_mod, "WrinkleDetector", _StubDetector)

    with TestClient(main_mod.app) as client:
        yield client


# --- Health / root / validation ------------------------------------------


def test_health(app_client: TestClient) -> None:
    r = app_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root(app_client: TestClient) -> None:
    r = app_client.get("/")
    assert r.status_code == 200
    body = r.json()
    for key in ("name", "version", "docs"):
        assert key in body, f"missing key {key!r} in root response"


def test_analyze_rejects_malformed_url(app_client: TestClient) -> None:
    r = app_client.post("/analyze", json={"image_url": "not-a-url"})
    # Pydantic HttpUrl rejects → FastAPI returns 422 by default.
    assert r.status_code == 422


def test_analyze_rejects_unsupported_scheme(app_client: TestClient) -> None:
    # Pydantic HttpUrl only allows http/https.
    r = app_client.post("/analyze", json={"image_url": "file:///etc/passwd"})
    assert r.status_code == 422


def test_analyze_rejects_missing_body(app_client: TestClient) -> None:
    r = app_client.post("/analyze", json={})
    assert r.status_code == 422


# --- SSRF guard ----------------------------------------------------------


def test_analyze_rejects_localhost(app_client: TestClient) -> None:
    r = app_client.post("/analyze", json={"image_url": "http://localhost/foo.jpg"})
    assert r.status_code == 400
    assert r.json().get("error_code") == "URLBlockedError"


def test_analyze_rejects_localhost_dot_local(app_client: TestClient) -> None:
    r = app_client.post("/analyze", json={"image_url": "http://router.local/foo.jpg"})
    assert r.status_code == 400
    assert r.json().get("error_code") == "URLBlockedError"


def test_analyze_rejects_private_ip(app_client: TestClient) -> None:
    r = app_client.post("/analyze", json={"image_url": "http://192.168.1.1/foo.jpg"})
    assert r.status_code == 400
    assert r.json().get("error_code") == "URLBlockedError"


def test_analyze_rejects_loopback_ip(app_client: TestClient) -> None:
    r = app_client.post("/analyze", json={"image_url": "http://127.0.0.1/foo.jpg"})
    assert r.status_code == 400
    assert r.json().get("error_code") == "URLBlockedError"


def test_analyze_rejects_link_local_ip(app_client: TestClient) -> None:
    # AWS-metadata IP — classic SSRF target.
    r = app_client.post("/analyze", json={"image_url": "http://169.254.169.254/latest/meta-data/"})
    assert r.status_code == 400
    assert r.json().get("error_code") == "URLBlockedError"


def test_analyze_rejects_ipv6_loopback(app_client: TestClient) -> None:
    # ``::1`` is the IPv6 loopback address. Bracketed in URLs.
    r = app_client.post("/analyze", json={"image_url": "http://[::1]/foo.jpg"})
    assert r.status_code == 400
    assert r.json().get("error_code") == "URLBlockedError"


# --- Distinct face-detection error codes ---------------------------------


def _stub_fetch_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``app.main.fetch_image`` with a stub that returns a tiny RGB image.

    The stub's return value never reaches the detector in these tests
    because we also stub ``detector.analyze``; we just need the route
    handler to get past the fetch step.
    """
    from app import main as main_mod

    async def fake_fetch(*_args, **_kwargs):
        return np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr(main_mod, "fetch_image", fake_fetch)


def test_analyze_returns_distinct_no_face_error_code(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.wrinkle_detector import NoFaceDetectedError

    _stub_fetch_image(monkeypatch)

    def raise_no_face(_img):
        raise NoFaceDetectedError("no face detected")

    monkeypatch.setattr(app_client.app.state.detector, "analyze", raise_no_face)

    r = app_client.post("/analyze", json={"image_url": "https://example.com/x.jpg"})
    assert r.status_code == 400
    body = r.json()
    assert body["error_code"] == "NoFaceDetectedError"
    assert "no face" in body["detail"].lower()


def test_analyze_returns_distinct_multiple_faces_error_code(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.wrinkle_detector import MultipleFacesDetectedError

    _stub_fetch_image(monkeypatch)

    def raise_multi(_img):
        raise MultipleFacesDetectedError("expected exactly one face, got 2")

    monkeypatch.setattr(app_client.app.state.detector, "analyze", raise_multi)

    r = app_client.post("/analyze", json={"image_url": "https://example.com/x.jpg"})
    assert r.status_code == 400
    body = r.json()
    assert body["error_code"] == "MultipleFacesDetectedError"
    # Confirm it's distinct from the no-face code so callers can branch.
    assert body["error_code"] != "NoFaceDetectedError"


# --- image_loader unit tests via httpx.MockTransport ---------------------
# These exercise the failure modes that hit production hardest: hostile
# servers that lie about size or content type. We bypass the SSRF guard
# (block_private_ips=False) since these tests are about HTTP behavior,
# not the SSRF guard, which is covered above.


def _patched_async_client_factory(handler: Callable[[httpx.Request], httpx.Response]) -> Callable:
    """Build a drop-in replacement for ``httpx.AsyncClient`` that uses MockTransport."""

    class _MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            # follow_redirects is irrelevant under MockTransport.
            super().__init__(*args, **kwargs)

    return _MockClient


def _run(coro):
    """Tiny helper so each test reads as a synchronous unit test."""
    return asyncio.run(coro)


def test_image_loader_rejects_non_image_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import HttpUrl

    from app.services import image_loader as il

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html/>")

    monkeypatch.setattr(il.httpx, "AsyncClient", _patched_async_client_factory(handler))

    with pytest.raises(il.InvalidContentTypeError):
        _run(
            il.fetch_image(
                HttpUrl("https://example.com/page"),
                timeout_s=5.0,
                max_size_bytes=15 * 1024 * 1024,
                block_private_ips=False,
            )
        )


def test_image_loader_size_cap_via_content_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cheap-reject path: server reports a Content-Length over the cap."""
    from pydantic import HttpUrl

    from app.services import image_loader as il

    BIG = b"\x00" * (16 * 1024 * 1024)  # 16 MB > 15 MB cap

    def handler(_request: httpx.Request) -> httpx.Response:
        # MockTransport will set Content-Length from `content` automatically.
        return httpx.Response(200, headers={"content-type": "image/png"}, content=BIG)

    monkeypatch.setattr(il.httpx, "AsyncClient", _patched_async_client_factory(handler))

    with pytest.raises(il.ImageTooLargeError):
        _run(
            il.fetch_image(
                HttpUrl("https://example.com/big.png"),
                timeout_s=5.0,
                max_size_bytes=15 * 1024 * 1024,
                block_private_ips=False,
            )
        )


def test_image_loader_size_cap_via_streaming_when_content_length_lies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming-cap path: server LIES about size, must still be caught.

    We hand back 16 MB of bytes but rewrite the Content-Length header to
    claim 100 bytes. The cheap-reject path should be satisfied (100 ≤ cap),
    so the stream-and-count branch must be the one that raises.
    """
    from pydantic import HttpUrl

    from app.services import image_loader as il

    BIG = b"\x00" * (16 * 1024 * 1024)

    def handler(_request: httpx.Request) -> httpx.Response:
        resp = httpx.Response(200, headers={"content-type": "image/png"}, content=BIG)
        # Lie. The body still contains 16 MB; only the header changed.
        resp.headers["content-length"] = "100"
        return resp

    monkeypatch.setattr(il.httpx, "AsyncClient", _patched_async_client_factory(handler))

    with pytest.raises(il.ImageTooLargeError):
        _run(
            il.fetch_image(
                HttpUrl("https://example.com/lying.png"),
                timeout_s=5.0,
                max_size_bytes=15 * 1024 * 1024,
                block_private_ips=False,
            )
        )


def test_image_loader_rejects_undecodable_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server returned image/* but the bytes aren't a real image."""
    from pydantic import HttpUrl

    from app.services import image_loader as il

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=b"this is not actually a PNG",
        )

    monkeypatch.setattr(il.httpx, "AsyncClient", _patched_async_client_factory(handler))

    with pytest.raises(il.ImageDecodeError):
        _run(
            il.fetch_image(
                HttpUrl("https://example.com/fake.png"),
                timeout_s=5.0,
                max_size_bytes=15 * 1024 * 1024,
                block_private_ips=False,
            )
        )


def test_image_loader_rejects_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-2xx response from upstream becomes URLUnreachableError → 400."""
    from pydantic import HttpUrl

    from app.services import image_loader as il

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, headers={"content-type": "text/plain"}, content=b"nope")

    monkeypatch.setattr(il.httpx, "AsyncClient", _patched_async_client_factory(handler))

    with pytest.raises(il.URLUnreachableError):
        _run(
            il.fetch_image(
                HttpUrl("https://example.com/missing.png"),
                timeout_s=5.0,
                max_size_bytes=15 * 1024 * 1024,
                block_private_ips=False,
            )
        )


# --- Severity & measure unit checks --------------------------------------
# Quick correctness checks on the parts of the pipeline that are pure
# functions of dicts (no images required). If these break, the JSON
# shape downstream is wrong before any HTTP layer gets involved.


def test_severity_grades_thresholds() -> None:
    from app.core.severity import compute_severity

    # Empty-ish zone → grade 0
    s = compute_severity({"density_mm_per_cm2": 0.0, "longest_wrinkle_mm": 0.0})
    assert s["grade"] == 0
    assert s["label"] == "None"
    assert s["driven_by"] == "n/a"

    # Density-driven moderate
    s = compute_severity({"density_mm_per_cm2": 5.0, "longest_wrinkle_mm": 5.0})
    assert s["density_grade"] == 2
    assert s["length_grade"] == 1
    assert s["grade"] == 2
    assert s["driven_by"] == "density"

    # Length-driven pronounced
    s = compute_severity({"density_mm_per_cm2": 1.0, "longest_wrinkle_mm": 25.0})
    assert s["length_grade"] == 3
    assert s["grade"] == 3
    assert s["driven_by"] == "length"

    # Tied at top → "both"
    s = compute_severity({"density_mm_per_cm2": 2.0, "longest_wrinkle_mm": 5.0})
    assert s["density_grade"] == 1
    assert s["length_grade"] == 1
    assert s["driven_by"] == "both"


def test_measure_zone_handles_empty_skeleton() -> None:
    from app.core.measure import measure_zone

    skel = np.zeros((100, 100), dtype=bool)
    mask = np.full((100, 100), 255, dtype=np.uint8)
    out = measure_zone(skel, mask, mm_per_px=0.25)
    assert out["wrinkle_count"] == 0
    assert out["total_length_mm"] == 0.0
    assert out["zone_area_cm2"] > 0
    assert out["component_lengths_mm"] == []
