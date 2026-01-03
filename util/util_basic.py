# src/config/util_basic.py
from __future__ import annotations

from typing import Any, Dict, Type


def validate_required_str(
    d: Dict[str, Any],
    path: str,
    exc: Type[Exception] = ValueError,
) -> str:
    """path 예: 'repos.internal.url'"""
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise exc(f"필수 항목 누락: {path}")
        cur = cur[key]
    if not isinstance(cur, str) or not cur.strip():
        raise exc(f"빈 문자열/타입 오류: {path}")
    return cur.strip()


def validate_optional_bool(
    d: Dict[str, Any],
    path: str,
    default: bool,
    exc: Type[Exception] = ValueError,
) -> bool:
    """
    path가 없으면 default 반환.
    있으면 bool 타입만 허용. (문자열 "false" 같은 값은 에러)
    """
    cur: Any = d
    keys = path.split(".")
    for key in keys[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]

    last = keys[-1]
    if not isinstance(cur, dict) or last not in cur:
        return default

    val = cur[last]
    if isinstance(val, bool):
        return val
    raise exc(f"타입 오류(bool 필요): {path}")


def validate_optional_str(
    d: Dict[str, Any],
    path: str,
    default: str,
    exc: Type[Exception] = ValueError,
) -> str:
    """
    path가 없으면 default 반환.
    있으면 str 타입만 허용.
    """
    cur: Any = d
    keys = path.split(".")
    for key in keys[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]

    last = keys[-1]
    if not isinstance(cur, dict) or last not in cur:
        return default

    val = cur[last]
    if isinstance(val, str):
        return val
    raise exc(f"타입 오류(str 필요): {path}")


def require_dict(
    d: Dict[str, Any],
    path: str,
    exc: Type[Exception] = ValueError,
) -> Dict[str, Any]:
    """
    path의 값이 dict인지 보장. 없거나(dict가 아니면) 예외.
    예: require_dict(cfg, "repos.internal")
    """
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise exc(f"필수 객체 누락: {path}")
        cur = cur[key]
    if not isinstance(cur, dict):
        raise exc(f"타입 오류(dict 필요): {path}")
    return cur
