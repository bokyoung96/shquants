from __future__ import annotations

from datetime import date
from math import isfinite


def read_date_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a YYYY-MM-DD date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a YYYY-MM-DD date string") from exc
    if value != parsed.isoformat():
        raise ValueError(f"{key} must be a YYYY-MM-DD date string")
    return value


def read_date_tuple(payload: dict[str, object], key: str, error_key: str) -> tuple[str, ...]:
    if key not in payload:
        return ()
    values = payload[key]
    if not isinstance(values, list):
        raise ValueError(f"{error_key} must be a list")
    parsed: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{error_key} entries must be YYYY-MM-DD date strings")
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{error_key} entries must be YYYY-MM-DD date strings") from exc
        if value != parsed_date.isoformat():
            raise ValueError(f"{error_key} entries must be YYYY-MM-DD date strings")
        parsed.append(value)
    return tuple(parsed)


def read_bool(payload: dict[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be a boolean")


def read_optional_bool(payload: dict[str, object], key: str, default: bool, error_key: str | None = None) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        return value
    label = error_key or key
    raise ValueError(f"{label} must be a boolean")


def read_object(payload: dict[str, object], key: str) -> dict[str, object] | None:
    if key not in payload:
        return None
    value = payload[key]
    if isinstance(value, dict):
        return value
    raise ValueError(f"{key} must be an object")


def read_required_string(payload: dict[str, object], key: str, error_key: str | None = None) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    label = error_key or key
    raise ValueError(f"{label} must be a string")


def read_optional_string(payload: dict[str, object], key: str, error_key: str | None = None) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        return None
    if isinstance(value, str):
        return value
    label = error_key or key
    raise ValueError(f"{label} must be a string")


def read_string_choice(
    payload: dict[str, object],
    key: str,
    *,
    default: str | None = None,
    error_key: str | None = None,
    allowed: set[str],
) -> str:
    if key not in payload:
        if default is None:
            return read_required_string(payload, key, error_key)
        value = default
    else:
        value = read_required_string(payload, key, error_key)
    if value not in allowed:
        label = error_key or key
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"{label} must be one of: {allowed_values}")
    return value


def read_int(
    payload: dict[str, object],
    key: str,
    default: int | None = None,
    *,
    error_key: str | None = None,
    min_value: int | None = None,
) -> int | None:
    if key not in payload:
        return default
    value = payload[key]
    label = error_key or key
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if min_value is not None and value < min_value:
        raise ValueError(f"{label} must be >= {min_value}")
    return value


def read_float(
    payload: dict[str, object],
    key: str,
    default: float | None = None,
    *,
    error_key: str | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    if key not in payload:
        return default
    value = payload[key]
    label = error_key or key
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{label} must be >= {min_value:g}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{label} must be <= {max_value:g}")
    return parsed


def read_params(payload: dict[str, object], key: str, error_key: str | None = None) -> dict[str, object]:
    if key not in payload:
        return {}
    value = payload[key]
    if isinstance(value, dict):
        return value
    label = error_key or key
    raise ValueError(f"{label} must be an object")
