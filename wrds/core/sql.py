from __future__ import annotations


def limit(sql: str, value: int | None) -> str:
    if value is None:
        return sql
    return f"{sql} limit {int(value)}"

