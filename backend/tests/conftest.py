import sys
import types


if "tenacity" not in sys.modules:
    class _RetryError(Exception):
        """Lightweight stand-in for tenacity.RetryError in tests."""

    def _retry(*args, **kwargs):  # noqa: D401 - simple stub
        def decorator(func):
            return func

        return decorator

    def _placeholder(*args, **kwargs):  # noqa: D401 - simple stub
        return None

    tenacity_stub = types.SimpleNamespace(
        RetryError=_RetryError,
        retry=_retry,
        retry_if_exception_type=_placeholder,
        stop_after_attempt=_placeholder,
        wait_random_exponential=_placeholder,
    )
    sys.modules["tenacity"] = tenacity_stub
