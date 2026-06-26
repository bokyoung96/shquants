from __future__ import annotations

from collections.abc import Callable
from typing import Any


Callback = Callable[..., None]


class Control:
    def __init__(self, raw: Any) -> None:
        self.raw = raw

    def start(self) -> None:
        self._call("SetQtMode", True)
        self._call("RunIndiPython")

    def state(self) -> int:
        return int(self._call("GetCommState"))

    def login(self, user: str, password: str, cert_password: str, starter: str) -> bool:
        return bool(self._call("StartIndi", user, password, cert_password, starter))

    def on(self, name: str, callback: Callback) -> None:
        if hasattr(self.raw, "SetCallBack"):
            try:
                self.raw.SetCallBack(name, callback)
                return
            except AttributeError:
                pass

        signal = getattr(self.raw, name, None)
        if signal is not None and hasattr(signal, "connect"):
            signal.connect(callback)
            return

        raise RuntimeError(f"control does not support callback: {name}")

    def query(self, name: str) -> None:
        self._call("SetQueryName", name)

    def set(self, index: int, value: object) -> None:
        self._call("SetSingleData", index, str(value))

    def request(self) -> int:
        return int(self._call("RequestData"))

    def request_rt(self, real_type: str | None = None, code: str | None = None) -> int:
        if real_type is None and code is None:
            return int(self._call("RequestRTReg"))
        return int(self._call("RequestRTReg", real_type, code))

    def stop_rt(self, real_type: str, code: str = "") -> None:
        self._call("UnRequestRTReg", real_type, code)

    def get(self, index: int) -> str:
        return str(self._call("GetSingleData", index))

    def _call(self, name: str, *args: object) -> Any:
        func = getattr(self.raw, name, None)
        if callable(func):
            return func(*args)

        dynamic = getattr(self.raw, "dynamicCall", None)
        if callable(dynamic):
            return dynamic(_signature(name, len(args)), *args)

        raise AttributeError(f"control does not support {name}")


def _signature(name: str, argc: int) -> str:
    signatures = {
        ("SetQtMode", 1): "SetQtMode(bool)",
        ("RunIndiPython", 0): "RunIndiPython()",
        ("GetCommState", 0): "GetCommState()",
        ("StartIndi", 4): "StartIndi(QString, QString, QString, QString)",
        ("SetQueryName", 1): "SetQueryName(QString)",
        ("SetSingleData", 2): "SetSingleData(int, QString)",
        ("RequestData", 0): "RequestData()",
        ("RequestRTReg", 0): "RequestRTReg()",
        ("RequestRTReg", 2): "RequestRTReg(QString, QString)",
        ("UnRequestRTReg", 2): "UnRequestRTReg(QString, QString)",
        ("GetSingleData", 1): "GetSingleData(int)",
    }
    return signatures.get((name, argc), f"{name}()")
