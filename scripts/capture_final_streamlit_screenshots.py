"""Capture four real headless-browser screenshots from saved Sprint 5 artifacts."""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _browser() -> Path:
    candidates = (
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    discovered = shutil.which("chrome") or shutil.which("msedge")
    if discovered:
        return Path(discovered)
    raise FileNotFoundError("Chrome or Microsoft Edge is required for screenshot capture")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _wait_for_server(url: str, *, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:  # noqa: S310
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"Streamlit did not become ready within {timeout_seconds} seconds")


def _read_json_url(url: str, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"DevTools endpoint returned a non-object payload: {url}")
    return payload


def _wait_for_debugger(port: int, *, timeout_seconds: float = 15.0) -> str:
    endpoint = f"http://127.0.0.1:{port}/json/version"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            payload = _read_json_url(endpoint)
            websocket_url = payload.get("webSocketDebuggerUrl")
            if isinstance(websocket_url, str) and websocket_url:
                return websocket_url
        except (OSError, json.JSONDecodeError, RuntimeError):
            time.sleep(0.2)
    raise TimeoutError("Headless browser DevTools endpoint did not become ready")


class _DevToolsClient:
    """Small dependency-free WebSocket client for the local Chrome DevTools API."""

    def __init__(self, websocket_url: str) -> None:
        parsed = urllib.parse.urlsplit(websocket_url)
        if parsed.scheme != "ws" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise RuntimeError("Refusing a non-local DevTools WebSocket endpoint")
        port = parsed.port or 80
        self._socket = socket.create_connection((parsed.hostname, port), timeout=10)
        self._socket.settimeout(30)
        self._buffer = bytearray()
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode("ascii")
        self._socket.sendall(request)
        response = bytearray()
        while b"\r\n\r\n" not in response:
            chunk = self._socket.recv(4096)
            if not chunk:
                raise RuntimeError("DevTools WebSocket closed during handshake")
            response.extend(chunk)
        header, remainder = bytes(response).split(b"\r\n\r\n", 1)
        if not header.startswith(b"HTTP/1.1 101"):
            raise RuntimeError(f"DevTools WebSocket handshake failed: {header!r}")
        self._buffer.extend(remainder)
        self._next_id = 1

    def close(self) -> None:
        try:
            self._send_frame(b"", opcode=0x8)
        except OSError:
            pass
        self._socket.close()

    def _read_exact(self, size: int) -> bytes:
        while len(self._buffer) < size:
            chunk = self._socket.recv(max(4096, size - len(self._buffer)))
            if not chunk:
                raise RuntimeError("DevTools WebSocket closed unexpectedly")
            self._buffer.extend(chunk)
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result

    def _send_frame(self, payload: bytes, *, opcode: int = 0x1) -> None:
        mask = os.urandom(4)
        length = len(payload)
        if length < 126:
            header = bytes((0x80 | opcode, 0x80 | length))
        elif length < 65_536:
            header = bytes((0x80 | opcode, 0x80 | 126)) + length.to_bytes(2, "big")
        else:
            header = bytes((0x80 | opcode, 0x80 | 127)) + length.to_bytes(8, "big")
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        self._socket.sendall(header + mask + masked)

    def _receive_text(self) -> str:
        fragments = bytearray()
        started = False
        while True:
            first, second = self._read_exact(2)
            finished = bool(first & 0x80)
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            length = second & 0x7F
            if length == 126:
                length = int.from_bytes(self._read_exact(2), "big")
            elif length == 127:
                length = int.from_bytes(self._read_exact(8), "big")
            mask = self._read_exact(4) if masked else b""
            payload = self._read_exact(length)
            if masked:
                payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
            if opcode == 0x8:
                raise RuntimeError("DevTools WebSocket closed unexpectedly")
            if opcode == 0x9:
                self._send_frame(payload, opcode=0xA)
                continue
            if opcode == 0x1:
                fragments = bytearray(payload)
                started = True
            elif opcode == 0x0 and started:
                fragments.extend(payload)
            else:
                continue
            if finished:
                return fragments.decode("utf-8")

    def command(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        command_id = self._next_id
        self._next_id += 1
        payload: dict[str, Any] = {"id": command_id, "method": method}
        if params is not None:
            payload["params"] = params
        if session_id is not None:
            payload["sessionId"] = session_id
        self._send_frame(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        while True:
            message = json.loads(self._receive_text())
            if message.get("id") != command_id:
                continue
            if "error" in message:
                raise RuntimeError(f"DevTools {method} failed: {message['error']}")
            result = message.get("result", {})
            if not isinstance(result, dict):
                raise RuntimeError(f"DevTools {method} returned an invalid result")
            return result


def _evaluate(client: _DevToolsClient, session_id: str, expression: str) -> Any:
    evaluated = client.command(
        "Runtime.evaluate",
        {"expression": expression, "returnByValue": True},
        session_id=session_id,
    )
    return evaluated.get("result", {}).get("value")


def _wait_for_text(
    client: _DevToolsClient,
    session_id: str,
    expected: str,
    *,
    timeout_seconds: float = 40.0,
) -> None:
    quoted = json.dumps(expected)
    expression = (
        "(() => {"
        "const text=(document.body && document.body.innerText)||'';"
        "const skeletons=document.querySelectorAll('[data-testid=stSkeleton]').length;"
        f"return {{ready:text.includes({quoted})&&skeletons===0,"
        "text:text.slice(0,500),skeletons:skeletons};"
        "})()"
    )
    deadline = time.monotonic() + timeout_seconds
    observed: Any = None
    while time.monotonic() < deadline:
        observed = _evaluate(client, session_id, expression)
        if isinstance(observed, dict) and observed.get("ready") is True:
            time.sleep(1.0)
            return
        time.sleep(0.4)
    raise TimeoutError(f"Streamlit page did not fully render: {expected}; {observed}")


def _click_text(client: _DevToolsClient, session_id: str, selector: str, label: str) -> None:
    expression = (
        "(() => {"
        f"const nodes=Array.from(document.querySelectorAll({json.dumps(selector)}));"
        f"const target=nodes.find(node=>(node.innerText||node.textContent||'').trim()==="
        f"{json.dumps(label)});"
        "if(!target){return false;}target.click();return true;"
        "})()"
    )
    if _evaluate(client, session_id, expression) is not True:
        raise RuntimeError(f"Could not click {label!r}")


def _fill_input(client: _DevToolsClient, session_id: str, label: str, value: str) -> None:
    expression = (
        "(() => {"
        "const input=Array.from(document.querySelectorAll('input')).find("
        f"node=>node.getAttribute('aria-label')==={json.dumps(label)});"
        "if(!input){return false;}"
        "const setter=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;"
        f"setter.call(input,{json.dumps(value)});"
        "input.dispatchEvent(new InputEvent('input',{bubbles:true,inputType:'insertText'}));"
        "input.dispatchEvent(new Event('change',{bubbles:true}));"
        "return true;})()"
    )
    if _evaluate(client, session_id, expression) is not True:
        raise RuntimeError(f"Could not fill {label!r}")
    time.sleep(0.5)


def _capture_rendered_page(
    client: _DevToolsClient,
    session_id: str,
    page_title: str,
    destination: Path,
) -> None:
    _wait_for_text(client, session_id, page_title)
    _evaluate(
        client,
        session_id,
        "document.querySelector('[data-testid=\"stMain\"]')?.scrollTo(0,0)",
    )
    time.sleep(0.5)
    layout = client.command("Page.getLayoutMetrics", session_id=session_id)
    content_size = layout.get("cssContentSize", {})
    content_width = max(1440.0, float(content_size.get("width", 1440.0)))
    content_height = max(1400.0, min(float(content_size.get("height", 1400.0)), 6000.0))
    captured = client.command(
        "Page.captureScreenshot",
        {
            "format": "png",
            "fromSurface": True,
            "captureBeyondViewport": True,
            "clip": {
                "x": 0,
                "y": 0,
                "width": content_width,
                "height": content_height,
                "scale": 1,
            },
        },
        session_id=session_id,
    )
    image_data = captured.get("data")
    if not isinstance(image_data, str) or not image_data:
        raise RuntimeError(f"DevTools did not return a screenshot for {page_title}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_bytes(base64.b64decode(image_data, validate=True))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/sprint5"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sprint5/screenshots"))
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    artifact_root = (project_root / arguments.artifact_root).resolve()
    output_dir = (project_root / arguments.output_dir).resolve()
    if not (artifact_root / "product" / "final_test_summary.json").is_file():
        raise FileNotFoundError("Verified Sprint 5 product artifacts are required")
    output_dir.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    environment = os.environ.copy()
    demo_email = f"screenshots-{secrets.token_hex(4)}@argus.example"
    demo_password = secrets.token_urlsafe(16)
    environment["ARGUS_ARTIFACT_DIR"] = str(artifact_root)
    environment["ARGUS_DEMO_EMAIL"] = demo_email
    environment["ARGUS_DEMO_PASSWORD"] = demo_password
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(project_root / "app.py"),
        "--server.headless=true",
        f"--server.port={port}",
        "--server.address=127.0.0.1",
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    server = subprocess.Popen(
        command,
        cwd=project_root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    browser = _browser()
    pages = {
        "executive_dashboard": "Overview",
        "investigation_queue": "Investigations",
        "case_investigator": "Case Investigator",
        "model_comparison": "Model Evidence",
    }
    created: list[Path] = []
    try:
        base_url = f"http://127.0.0.1:{port}"
        _wait_for_server(base_url)
        with tempfile.TemporaryDirectory(
            prefix="argus_s5_browser_", ignore_cleanup_errors=True
        ) as browser_profile:
            debugger_port = _free_port()
            browser_process = subprocess.Popen(
                [
                    str(browser),
                    "--headless=new",
                    "--disable-gpu",
                    "--disable-extensions",
                    "--hide-scrollbars",
                    "--no-first-run",
                    "--remote-allow-origins=*",
                    f"--remote-debugging-port={debugger_port}",
                    f"--user-data-dir={browser_profile}",
                    "--window-size=1440,1400",
                    "about:blank",
                ],
                cwd=project_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
            client: _DevToolsClient | None = None
            target_id: str | None = None
            try:
                client = _DevToolsClient(_wait_for_debugger(debugger_port))
                created_target = client.command("Target.createTarget", {"url": "about:blank"})
                target_id = str(created_target["targetId"])
                attached = client.command(
                    "Target.attachToTarget", {"targetId": target_id, "flatten": True}
                )
                session_id = str(attached["sessionId"])
                client.command("Page.enable", session_id=session_id)
                client.command("Runtime.enable", session_id=session_id)
                client.command(
                    "Emulation.setDeviceMetricsOverride",
                    {
                        "width": 1440,
                        "height": 1400,
                        "deviceScaleFactor": 1,
                        "mobile": False,
                    },
                    session_id=session_id,
                )
                client.command(
                    "Page.navigate", {"url": f"{base_url}/?view=login"}, session_id=session_id
                )
                _wait_for_text(client, session_id, "Corporate Login")
                _fill_input(client, session_id, "Corporate email", demo_email)
                _fill_input(client, session_id, "Password", demo_password)
                _click_text(client, session_id, "button", "Sign in")
                _wait_for_text(client, session_id, "Overview")
                for slug, page in pages.items():
                    if page != "Overview":
                        _click_text(client, session_id, "label", page)
                    destination = (output_dir / f"{slug}.png").resolve()
                    _capture_rendered_page(client, session_id, page, destination)
                    created.append(destination)
            finally:
                if client is not None and target_id is not None:
                    client.command("Target.closeTarget", {"targetId": target_id})
                if client is not None:
                    client.close()
                browser_process.terminate()
                try:
                    browser_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    browser_process.kill()
                    browser_process.wait(timeout=5)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
    print("Final Streamlit screenshots: PASS")
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
