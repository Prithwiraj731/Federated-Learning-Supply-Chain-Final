import base64
import hashlib
import hmac
import json
import os
import re
import threading
import time
from json import JSONDecodeError
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, parse, request

from cryptography.fernet import Fernet, InvalidToken


DEFAULT_SHARED_SECRET = "change-this-shared-secret-before-production"
DEFAULT_TIMEOUT = 3.0
ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".xls", ".xlsx"}
_SERVER_INSTANCES: Dict[int, Dict[str, Any]] = {}
_SERVER_LOCK = threading.Lock()
CONTROL_TOWER_STATE_PATH = os.path.join("sc50_logs", "control_tower_endpoint.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_shared_secret() -> str:
    return os.getenv("SC_SHARED_SECRET", DEFAULT_SHARED_SECRET)


def load_control_tower_endpoint() -> Optional[Dict[str, Any]]:
    if not os.path.exists(CONTROL_TOWER_STATE_PATH):
        return None
    with open(CONTROL_TOWER_STATE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_control_tower_endpoint(host: str, port: int) -> None:
    os.makedirs(os.path.dirname(CONTROL_TOWER_STATE_PATH), exist_ok=True)
    with open(CONTROL_TOWER_STATE_PATH, "w", encoding="utf-8") as handle:
        json.dump({"host": host, "port": port, "base_url": f"http://{host}:{port}"}, handle, indent=2)


def secure_filename(filename: str) -> str:
    base_name = os.path.basename(filename or "upload")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name).strip("._")
    return sanitized or "upload"


def validate_upload_filename(filename: str) -> str:
    safe_name = secure_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError("Only .csv, .xls, and .xlsx files are allowed.")
    return safe_name


def _canonical_json(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sign_payload(payload: Dict[str, Any], secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), _canonical_json(payload), hashlib.sha256).hexdigest()


def _derive_fernet_key(secret: str) -> bytes:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_json(payload: Dict[str, Any], secret: Optional[str] = None) -> str:
    secret = secret or get_shared_secret()
    cipher = Fernet(_derive_fernet_key(secret))
    return cipher.encrypt(_canonical_json(payload)).decode("utf-8")


def decrypt_json(token: str, secret: Optional[str] = None) -> Dict[str, Any]:
    secret = secret or get_shared_secret()
    cipher = Fernet(_derive_fernet_key(secret))
    try:
        raw = cipher.decrypt(token.encode("utf-8"))
    except InvalidToken as exc:
        raise ValueError("Unable to decrypt secure payload. Check the shared secret.") from exc
    return json.loads(raw.decode("utf-8"))


def build_envelope(
    payload: Dict[str, Any],
    purpose: str,
    sender: str,
    secret: Optional[str] = None,
) -> Dict[str, Any]:
    secret = secret or get_shared_secret()
    envelope = {
        "sender": sender,
        "purpose": purpose,
        "timestamp": int(time.time()),
        "ciphertext": encrypt_json(payload, secret),
    }
    envelope["signature"] = _sign_payload(envelope, secret)
    return envelope


def open_envelope(
    envelope: Dict[str, Any],
    secret: Optional[str] = None,
    max_age_seconds: int = 900,
) -> Dict[str, Any]:
    secret = secret or get_shared_secret()
    signature = envelope.get("signature", "")
    signed_payload = {
        "sender": envelope.get("sender", ""),
        "purpose": envelope.get("purpose", ""),
        "timestamp": envelope.get("timestamp", 0),
        "ciphertext": envelope.get("ciphertext", ""),
    }
    expected_signature = _sign_payload(signed_payload, secret)
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("Invalid secure signature.")

    timestamp = int(envelope.get("timestamp", 0))
    if abs(time.time() - timestamp) > max_age_seconds:
        raise ValueError("Secure payload expired.")

    return decrypt_json(envelope["ciphertext"], secret)


def _build_health_headers(sender: str, secret: str) -> Dict[str, str]:
    timestamp = str(int(time.time()))
    signed_payload = {"sender": sender, "purpose": "health", "timestamp": timestamp}
    signature = _sign_payload(signed_payload, secret)
    return {
        "X-SC-Sender": sender,
        "X-SC-Timestamp": timestamp,
        "X-SC-Signature": signature,
    }


def _verify_health_headers(headers, secret: str, max_age_seconds: int = 30) -> bool:
    sender = headers.get("X-SC-Sender", "")
    timestamp = headers.get("X-SC-Timestamp", "")
    signature = headers.get("X-SC-Signature", "")
    if not sender or not timestamp or not signature:
        return False
    try:
        timestamp_int = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - timestamp_int) > max_age_seconds:
        return False
    payload = {"sender": sender, "purpose": "health", "timestamp": timestamp}
    expected_signature = _sign_payload(payload, secret)
    return hmac.compare_digest(signature, expected_signature)


def _json_request(
    url: str,
    method: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    body = None
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, method=method, headers=request_headers)
    with request.urlopen(req, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        raw = response.read().decode("utf-8")
    try:
        return json.loads(raw) if raw else {}
    except JSONDecodeError as exc:
        raise ValueError(
            json.dumps(
                {
                    "error": "invalid_json",
                    "content_type": content_type,
                    "body_preview": raw[:240],
                }
            )
        ) from exc


def _friendly_endpoint_error(exc: Exception) -> str:
    raw_message = str(exc)
    try:
        parsed = json.loads(raw_message)
    except Exception:
        return raw_message

    if parsed.get("error") != "invalid_json":
        return raw_message

    content_type = str(parsed.get("content_type", "")).lower()
    body_preview = str(parsed.get("body_preview", "")).lower()
    looks_like_html = "<!doctype html" in body_preview or "<html" in body_preview
    looks_like_streamlit = "streamlit" in body_preview or "st-emotion-cache" in body_preview or "_stcore" in body_preview

    if "text/html" in content_type and (looks_like_html or looks_like_streamlit):
        return "This looks like a Streamlit web page, not a secure node endpoint. Use the node port for `/health` and `/sync`, not the browser UI port."
    if "text/html" in content_type:
        return "This endpoint returned HTML instead of secure node JSON. Check that the base URL points to the node endpoint rather than a web UI."
    return f"Endpoint returned non-JSON content ({content_type or 'unknown type'})."


def ping_node(
    base_url: str,
    sender: str = "control-tower",
    secret: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    secret = secret or get_shared_secret()
    headers = _build_health_headers(sender, secret)
    start = time.perf_counter()
    health_url = parse.urljoin(base_url.rstrip("/") + "/", "health")
    try:
        payload = _json_request(health_url, method="GET", headers=headers, timeout=timeout)
        latency_ms = (time.perf_counter() - start) * 1000
        payload["online"] = True
        payload["latency_ms"] = round(latency_ms, 1)
        payload["last_seen"] = utc_now_iso()
        return payload
    except error.HTTPError as exc:
        return {
            "online": False,
            "error": f"HTTP {exc.code}",
            "last_seen": None,
            "latency_ms": None,
        }
    except Exception as exc:
        return {
            "online": False,
            "error": _friendly_endpoint_error(exc),
            "last_seen": None,
            "latency_ms": None,
        }


def sync_bundle(
    base_url: str,
    bundle: Dict[str, Any],
    sender: str = "control-tower",
    secret: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    secret = secret or get_shared_secret()
    sync_url = parse.urljoin(base_url.rstrip("/") + "/", "sync")
    envelope = build_envelope(bundle, purpose="client-market-sync", sender=sender, secret=secret)
    return _json_request(sync_url, method="POST", payload=envelope, timeout=timeout)


def send_dataset_upload(
    base_url: str,
    client_id: str,
    file_name: str,
    file_bytes: bytes,
    sender: str = "client-node",
    secret: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    secret = secret or get_shared_secret()
    upload_url = parse.urljoin(base_url.rstrip("/") + "/", "upload")
    safe_name = validate_upload_filename(file_name)
    payload = {
        "client_id": str(client_id),
        "file_name": safe_name,
        "file_b64": base64.b64encode(file_bytes).decode("utf-8"),
        "size_bytes": len(file_bytes),
    }
    envelope = build_envelope(payload, purpose="raw-dataset-upload", sender=sender, secret=secret)
    return _json_request(upload_url, method="POST", payload=envelope, timeout=timeout)


def _bundle_file_path(bundle_dir: str) -> Path:
    return Path(bundle_dir) / "latest_bundle.secure.json"


def load_secure_bundle(bundle_dir: str, secret: Optional[str] = None) -> Optional[Dict[str, Any]]:
    bundle_path = _bundle_file_path(bundle_dir)
    if not bundle_path.exists():
        return None
    envelope = json.loads(bundle_path.read_text(encoding="utf-8"))
    return open_envelope(envelope, secret=secret, max_age_seconds=60 * 60 * 24 * 30)


class _SecureNodeHandler(BaseHTTPRequestHandler):
    server_version = "SecureSupplyChainNode/1.0"

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/health":
            self._send_json(404, {"error": "Not found"})
            return

        if not _verify_health_headers(self.headers, self.server.shared_secret):
            self._send_json(403, {"error": "Unauthorized"})
            return

        self.server.state["last_seen"] = utc_now_iso()
        response = {
            "node_id": self.server.state["node_id"],
            "node_name": self.server.state["node_name"],
            "client_id": self.server.state["client_id"],
            "status": "online",
            "last_seen": self.server.state["last_seen"],
            "last_sync_at": self.server.state.get("last_sync_at"),
            "bundle_path": str(_bundle_file_path(self.server.state["bundle_dir"])),
        }
        self._send_json(200, response)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/sync":
            self._send_json(404, {"error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        try:
            envelope = json.loads(raw_body)
            payload = open_envelope(envelope, secret=self.server.shared_secret)
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
            return

        recipient_id = str(payload.get("client", {}).get("client_id", ""))
        if recipient_id != self.server.state["client_id"]:
            self._send_json(409, {"ok": False, "error": "Payload recipient does not match node identity."})
            return

        bundle_dir = Path(self.server.state["bundle_dir"])
        bundle_dir.mkdir(parents=True, exist_ok=True)
        _bundle_file_path(str(bundle_dir)).write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        self.server.state["last_sync_at"] = utc_now_iso()
        self.server.state["last_seen"] = self.server.state["last_sync_at"]
        self._send_json(
            200,
            {
                "ok": True,
                "node_id": self.server.state["node_id"],
                "client_id": self.server.state["client_id"],
                "last_sync_at": self.server.state["last_sync_at"],
            },
        )


class _SecureNodeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class, shared_secret: str, state: Dict[str, Any]):
        super().__init__(server_address, handler_class)
        self.shared_secret = shared_secret
        self.state = state


class _ControlTowerHandler(BaseHTTPRequestHandler):
    server_version = "SecureControlTower/1.0"

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/health":
            self._send_json(404, {"error": "Not found"})
            return

        if not _verify_health_headers(self.headers, self.server.shared_secret):
            self._send_json(403, {"error": "Unauthorized"})
            return

        self.server.state["last_seen"] = utc_now_iso()
        self._send_json(
            200,
            {
                "role": "control-tower",
                "status": "online",
                "last_seen": self.server.state["last_seen"],
                "dataset_dir": self.server.state["dataset_dir"],
                "uploads_received": self.server.state.get("uploads_received", 0),
            },
        )

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/upload":
            self._send_json(404, {"error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        try:
            envelope = json.loads(raw_body)
            payload = open_envelope(envelope, secret=self.server.shared_secret)
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
            return

        client_id = str(payload.get("client_id", "")).strip()
        file_name = str(payload.get("file_name", "")).strip()
        file_b64 = str(payload.get("file_b64", "")).strip()
        if not client_id or not file_name or not file_b64:
            self._send_json(400, {"ok": False, "error": "Upload payload missing required fields."})
            return

        try:
            safe_name = validate_upload_filename(file_name)
            file_bytes = base64.b64decode(file_b64.encode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
            return

        suffix = Path(safe_name).suffix.lower()
        dataset_dir = Path(self.server.state["dataset_dir"])
        dataset_dir.mkdir(parents=True, exist_ok=True)
        target_path = dataset_dir / f"client_{client_id}__upload__current{suffix}"
        target_path.write_bytes(file_bytes)

        self.server.state["last_seen"] = utc_now_iso()
        self.server.state["last_upload_at"] = self.server.state["last_seen"]
        self.server.state["uploads_received"] = int(self.server.state.get("uploads_received", 0)) + 1
        self.server.state["last_upload"] = {
            "client_id": client_id,
            "file_name": safe_name,
            "saved_path": str(target_path),
            "size_bytes": len(file_bytes),
            "received_at": self.server.state["last_upload_at"],
        }
        self._send_json(
            200,
            {
                "ok": True,
                "client_id": client_id,
                "saved_path": str(target_path),
                "received_at": self.server.state["last_upload_at"],
                "size_bytes": len(file_bytes),
            },
        )


class _ControlTowerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class, shared_secret: str, state: Dict[str, Any]):
        super().__init__(server_address, handler_class)
        self.shared_secret = shared_secret
        self.state = state


def ensure_node_server(
    node_id: str,
    node_name: str,
    client_id: str,
    port: int,
    bundle_dir: str,
    host: str = "127.0.0.1",
    secret: Optional[str] = None,
) -> Dict[str, Any]:
    secret = secret or get_shared_secret()
    with _SERVER_LOCK:
        instance = _SERVER_INSTANCES.get(port)
        if instance:
            return {"ok": True, "already_running": True, "state": instance["server"].state}

        state = {
            "node_id": node_id,
            "node_name": node_name,
            "client_id": client_id,
            "bundle_dir": bundle_dir,
            "host": host,
            "port": port,
            "last_seen": None,
            "last_sync_at": None,
        }
        try:
            server = _SecureNodeHTTPServer((host, port), _SecureNodeHandler, shared_secret=secret, state=state)
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

        thread = threading.Thread(target=server.serve_forever, daemon=True, name=f"secure-node-{port}")
        thread.start()
        _SERVER_INSTANCES[port] = {"server": server, "thread": thread}
        return {"ok": True, "already_running": False, "state": state}


def get_node_server_state(port: int) -> Optional[Dict[str, Any]]:
    instance = _SERVER_INSTANCES.get(port)
    if not instance:
        return None
    return dict(instance["server"].state)


def ensure_control_tower_server(
    port: int,
    dataset_dir: str,
    host: str = "127.0.0.1",
    secret: Optional[str] = None,
) -> Dict[str, Any]:
    secret = secret or get_shared_secret()
    with _SERVER_LOCK:
        instance = _SERVER_INSTANCES.get(port)
        if instance:
            save_control_tower_endpoint(host, port)
            return {"ok": True, "already_running": True, "state": instance["server"].state}

        state = {
            "role": "control-tower",
            "dataset_dir": dataset_dir,
            "host": host,
            "port": port,
            "last_seen": None,
            "last_upload_at": None,
            "uploads_received": 0,
            "last_upload": None,
        }
        try:
            server = _ControlTowerHTTPServer((host, port), _ControlTowerHandler, shared_secret=secret, state=state)
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

        thread = threading.Thread(target=server.serve_forever, daemon=True, name=f"secure-control-tower-{port}")
        thread.start()
        _SERVER_INSTANCES[port] = {"server": server, "thread": thread}
        save_control_tower_endpoint(host, port)
        return {"ok": True, "already_running": False, "state": state}


def get_control_tower_server_state(port: int) -> Optional[Dict[str, Any]]:
    instance = _SERVER_INSTANCES.get(port)
    if not instance:
        return None
    return dict(instance["server"].state)
