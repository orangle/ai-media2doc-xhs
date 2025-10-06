from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import mimetypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence

import requests
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

LOGGER = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.iflow.cn/v1/chat/completions"
API_URL = os.getenv("IFLOW_API_URL", DEFAULT_API_URL)

MAX_WORKERS = max(1, int(os.getenv("IFLOW_MAX_WORKERS", os.getenv("MAX_WORKERS", "4"))))
_CACHE_DIR = Path(os.getenv("IFLOW_CACHE_DIR", ".cache")).expanduser()
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_LOCK = threading.Lock()

_SESSION = requests.Session()
_PROXIES: Dict[str, str] = {}
for scheme in ("http", "https"):
    env_key = f"{scheme.upper()}_PROXY"
    if os.getenv(env_key):
        _PROXIES[scheme] = os.getenv(env_key, "")
if _PROXIES:
    _SESSION.proxies.update(_PROXIES)

_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)


class IFlowRetryableError(Exception):
    """Errors that should trigger a retry."""


def _ensure_api_key() -> str:
    api_key = os.getenv("IFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("IFLOW_API_KEY is not set.")
    return api_key


def _hash_messages(messages: Sequence[Any]) -> str:
    normalized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _hash_images(images: Sequence[str]) -> str:
    if not images:
        return ""
    digest = hashlib.sha1()
    for entry in images:
        digest.update(entry.encode("utf-8"))
    return digest.hexdigest()


def _cache_path(cache_key: str) -> Path:
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{digest}.json"


def _load_cache(cache_key: str) -> Dict[str, Any] | None:
    path = _cache_path(cache_key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as cache_file:
            return json.load(cache_file)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read cache %s: %s", path, exc)
        return None


def _store_cache(cache_key: str, data: Dict[str, Any]) -> None:
    path = _cache_path(cache_key)
    try:
        with path.open("w", encoding="utf-8") as cache_file:
            json.dump(data, cache_file, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to write cache %s: %s", path, exc)


def _infer_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/jpeg"


def _encode_image(entry: Any) -> tuple[str, str]:
    if isinstance(entry, dict):
        path = entry.get("path") or entry.get("image_path")
        data_url = entry.get("data") or entry.get("image_url")
    else:
        path = entry
        data_url = None

    if isinstance(path, str) and path.startswith("data:"):
        data_url = path
        path = None

    if data_url and not isinstance(data_url, str):
        data_url = str(data_url)

    if data_url:
        try:
            base64_part = data_url.split(",", 1)[1]
            raw_bytes = base64.b64decode(base64_part)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Invalid data URL provided for vision call") from exc
        sha1 = hashlib.sha1(raw_bytes).hexdigest()
        return data_url, sha1

    if not path:
        raise ValueError("Image entry must contain a path or data URL")

    file_path = Path(path).expanduser().resolve()
    with open(file_path, "rb") as image_file:
        image_bytes = image_file.read()
    sha1 = hashlib.sha1(image_bytes).hexdigest()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = _infer_mime(file_path)
    return f"data:{mime};base64,{b64}", sha1


def _prepare_messages(messages: Sequence[Any], image_payloads: Sequence[str]) -> List[Dict[str, Any]]:
    prepared = copy.deepcopy(messages)
    image_iter = iter(image_payloads)

    for message in prepared:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "input_image":
                if "image_url" in part and not part.get("image_url", "").startswith("data:"):
                    # Normalize unexpected URLs by skipping caching but still send as-is.
                    continue
                try:
                    image_data_url = next(image_iter)
                except StopIteration:
                    raise ValueError("Number of images does not match message placeholders") from None
                part.pop("image_path", None)
                part["image_url"] = image_data_url
    remaining = list(image_iter)
    if remaining:
        raise ValueError("More images provided than placeholders in messages")
    return prepared


def _submit_request(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {_ensure_api_key()}",
        "Content-Type": "application/json",
    }

    start = time.monotonic()
    try:
        response = _SESSION.post(API_URL, headers=headers, data=json.dumps(payload), timeout=timeout_s)
    except requests.Timeout as exc:
        LOGGER.warning("iFlow request timeout after %.1fs for model=%s", timeout_s, payload.get("model"))
        raise IFlowRetryableError("Request timeout") from exc
    except requests.RequestException as exc:  # noqa: BLE001
        LOGGER.error("iFlow request network error for model=%s: %s", payload.get("model"), exc)
        raise IFlowRetryableError("Network error") from exc

    duration = time.monotonic() - start
    status = response.status_code

    if status in {429} or 500 <= status < 600:
        LOGGER.warning("iFlow request retryable status=%s model=%s duration=%.2fs", status, payload.get("model"), duration)
        raise IFlowRetryableError(f"Retryable HTTP {status}")

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        LOGGER.error(
            "iFlow request failed status=%s model=%s duration=%.2fs", status, payload.get("model"), duration
        )
        raise

    try:
        data = response.json()
    except ValueError as exc:  # noqa: BLE001
        LOGGER.error("iFlow returned non-JSON response for model=%s", payload.get("model"))
        raise RuntimeError("Invalid JSON from iFlow") from exc

    LOGGER.info(
        "iFlow request success model=%s status=%s duration=%.2fs", payload.get("model"), status, duration
    )
    return data


@retry(
    reraise=True,
    retry=retry_if_exception_type(IFlowRetryableError),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=20),
)
def _request_with_retry(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    return _submit_request(payload, timeout_s)


def _execute_with_pool(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    future = _EXECUTOR.submit(_request_with_retry, payload, timeout_s)
    try:
        return future.result()
    except RetryError as exc:
        raise exc.last_attempt.exception()  # type: ignore[misc]


def _chat_common(
    model: str,
    payload_messages: Sequence[Any],
    timeout_s: float,
    *,
    cache_messages: Sequence[Any] | None = None,
    extra_cache_key: str = "",
    **payload_overrides: Any,
) -> Dict[str, Any]:
    cache_basis = cache_messages if cache_messages is not None else payload_messages
    prompt_hash = _hash_messages(cache_basis)
    overrides_hash = ""
    if payload_overrides:
        try:
            overrides_hash = hashlib.sha1(
                json.dumps(payload_overrides, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        except TypeError:
            overrides_hash = hashlib.sha1(str(sorted(payload_overrides.items())).encode("utf-8")).hexdigest()
    cache_key = f"{model}:{prompt_hash}:{extra_cache_key}:{overrides_hash}"

    with _CACHE_LOCK:
        cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    payload: Dict[str, Any] = {
        "model": model,
        "messages": copy.deepcopy(payload_messages),
    }
    payload.update(payload_overrides)

    data = _execute_with_pool(payload, timeout_s)

    with _CACHE_LOCK:
        _store_cache(cache_key, data)

    return data


def chat_completion(model: str, messages: Sequence[Any], timeout_s: float = 30, **payload_overrides: Any) -> Dict[str, Any]:
    """Call iFlow text chat completion with retries and caching."""
    return _chat_common(model, messages, timeout_s, **payload_overrides)


def chat_vision(
    model: str,
    messages: Sequence[Any],
    images: Sequence[Any],
    timeout_s: float = 45,
    **payload_overrides: Any,
) -> Dict[str, Any]:
    """Call iFlow vision chat endpoint with retries, concurrency control, and caching."""
    image_payloads: List[str] = []
    image_hash_parts: List[str] = []
    for image in images:
        data_url, sha1 = _encode_image(image)
        image_payloads.append(data_url)
        image_hash_parts.append(sha1)

    sanitized_messages = copy.deepcopy(messages)
    for message in sanitized_messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "input_image":
                if "image_path" in part:
                    part.pop("image_path", None)
                if "image_url" in part and isinstance(part["image_url"], str) and part["image_url"].startswith("data:"):
                    part["image_url"] = "__IMAGE__"

    extra_key = _hash_images(image_hash_parts)

    payload_messages = _prepare_messages(messages, image_payloads)

    return _chat_common(
        model,
        payload_messages,
        timeout_s,
        cache_messages=sanitized_messages,
        extra_cache_key=extra_key,
        **payload_overrides,
    )


def clear_cache(prefix: str | None = None) -> int:
    """Clear cached responses. Returns the number of files removed."""
    removed = 0
    with _CACHE_LOCK:
        for path in _CACHE_DIR.glob("*.json"):
            if prefix and not path.name.startswith(prefix):
                continue
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                continue
    return removed


def get_runtime_config() -> Dict[str, Any]:
    return {
        "api_url": API_URL,
        "max_workers": MAX_WORKERS,
        "retries": 3,
        "cache_dir": str(_CACHE_DIR),
    }
