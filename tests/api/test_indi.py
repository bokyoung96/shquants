from __future__ import annotations

import json
from pathlib import Path

import pytest

from api import Order, make
from api.client import Indi
from api.config import load_config


class FakeControl:
    def __init__(self, state: int = 0) -> None:
        self.state = state
        self.calls: list[tuple] = []
        self.callbacks = {}
        self.single = {}
        self.rqid = 100

    def SetQtMode(self, flag):
        self.calls.append(("SetQtMode", flag))
        return True

    def RunIndiPython(self):
        self.calls.append(("RunIndiPython",))
        return True

    def GetCommState(self):
        self.calls.append(("GetCommState",))
        return self.state

    def StartIndi(self, user, password, cert_password, starter):
        self.calls.append(("StartIndi", user, password, cert_password, starter))
        self.state = 0
        return True

    def SetCallBack(self, name, callback):
        self.callbacks[name] = callback
        return True

    def SetQueryName(self, name):
        self.calls.append(("SetQueryName", name))
        return True

    def SetSingleData(self, index, value):
        self.calls.append(("SetSingleData", index, value))
        return True

    def RequestData(self):
        self.rqid += 1
        self.calls.append(("RequestData", self.rqid))
        return self.rqid

    def RequestRTReg(self, real_type=None, code=None):
        self.rqid += 1
        self.calls.append(("RequestRTReg", real_type, code, self.rqid))
        return self.rqid

    def UnRequestRTReg(self, real_type, code):
        self.calls.append(("UnRequestRTReg", real_type, code))
        return True

    def GetSingleData(self, index):
        return self.single.get(index, "")


class Signal:
    def __init__(self) -> None:
        self.callback = None

    def connect(self, callback):
        self.callback = callback


class SignalControl(FakeControl):
    def __init__(self, state: int = 0) -> None:
        super().__init__(state)
        self.ReceiveData = Signal()
        self.ReceiveSysMsg = Signal()
        self.ReceiveRTData = Signal()

    def SetCallBack(self, name, callback):
        raise AttributeError(name)


class NoCallbackControl(FakeControl):
    def SetCallBack(self, name, callback):
        raise AttributeError(name)


def make_client(tmp_path: Path, state: int = 0):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "user": "uid",
                "password": "pw",
                "cert_password": "cert",
                "account": "1234567890",
                "account_password": "0000",
                "starter": "C:/SHINHAN-i/indi/GiExpertStarter.exe",
            }
        ),
        encoding="utf-8",
    )
    tr = FakeControl(state=state)
    real = FakeControl(state=state)
    return make(config_path, controls=lambda: (tr, real)), tr, real


def test_load_config_keeps_credentials_in_local_json(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"account": "1111", "account_password": "2222"}),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.account == "1111"
    assert config.account_password == "2222"
    assert config.starter.endswith("GiExpertStarter.exe")


def test_connect_reuses_existing_indi_login(tmp_path: Path):
    client, tr, real = make_client(tmp_path, state=0)

    assert client.connect() is True

    assert ("GetCommState",) in tr.calls
    assert not [call for call in tr.calls if call[0] == "StartIndi"]
    assert ("RunIndiPython",) in tr.calls
    assert ("RunIndiPython",) in real.calls


def test_connect_starts_indi_when_session_is_closed(tmp_path: Path):
    client, tr, _real = make_client(tmp_path, state=1)

    assert client.connect() is True

    assert (
        "StartIndi",
        "uid",
        "pw",
        "cert",
        "C:/SHINHAN-i/indi/GiExpertStarter.exe",
    ) in tr.calls


def test_subscribe_quote_registers_sc_pipeline_and_parses_tick(tmp_path: Path):
    client, tr, real = make_client(tmp_path)
    ticks = []

    rqid = client.subscribe_quote("005930", ticks.append)

    assert ("SetQueryName", "SC") in tr.calls
    assert ("SetSingleData", 0, "005930") in tr.calls
    assert rqid in client.requests

    tr.single = {
        0: "KR7005930003",
        1: "005930",
        2: "093001",
        3: "71000",
        7: "1200",
        8: "85200000",
        9: "10",
        10: "70000",
        11: "71500",
        12: "69900",
    }
    client.on_data(tr, rqid)

    assert ticks[-1].code == "005930"
    assert ticks[-1].price == 71000
    assert ("RequestRTReg", "SC", "005930", real.rqid) in real.calls


def test_unsubscribe_before_quote_ack_does_not_arm_realtime(tmp_path: Path):
    client, tr, real = make_client(tmp_path)
    ticks = []

    rqid = client.subscribe_quote("005930", ticks.append)
    client.unsubscribe_quote("005930")
    tr.single = {1: "005930", 2: "093001", 3: "71000"}
    client.on_data(tr, rqid)

    assert ticks == []
    assert not [call for call in real.calls if call[0] == "RequestRTReg"]


def test_order_sets_saba101u1_fields_with_short_names(tmp_path: Path):
    client, tr, _real = make_client(tmp_path)

    rqid = client.buy("005930", qty=3, price=71000, hoga="0")

    assert rqid in client.requests
    assert ("SetQueryName", "SABA101U1") in tr.calls
    assert ("SetSingleData", 0, "1234567890") in tr.calls
    assert ("SetSingleData", 2, "0000") in tr.calls
    assert ("SetSingleData", 7, "2") in tr.calls
    assert ("SetSingleData", 8, "A005930") in tr.calls
    assert ("SetSingleData", 9, "3") in tr.calls
    assert ("SetSingleData", 10, "71000") in tr.calls
    assert ("SetSingleData", 12, "0") in tr.calls


def test_order_can_use_explicit_account_without_config(tmp_path: Path):
    client, tr, _real = make_client(tmp_path)
    order = Order(
        code="000660",
        qty=2,
        side="sell",
        account="acc",
        account_password="pwd",
        hoga="1",
    )

    client.order(order)

    assert ("SetSingleData", 0, "acc") in tr.calls
    assert ("SetSingleData", 2, "pwd") in tr.calls
    assert ("SetSingleData", 7, "1") in tr.calls


@pytest.mark.parametrize(
    "order,error",
    [
        (Order(code="", qty=1, side="buy"), "code"),
        (Order(code="005930", qty=0, side="buy"), "qty"),
        (Order(code="005930", qty=1, side="buy", hoga="bad"), "hoga"),
        (Order(code="005930", qty=1, side="buy", market="bad"), "market"),
        (Order(code="005930", qty=1, side="buy", condition="bad"), "condition"),
    ],
)
def test_order_rejects_invalid_live_inputs(tmp_path: Path, order: Order, error: str):
    client, tr, _real = make_client(tmp_path)

    with pytest.raises(ValueError, match=error):
        client.order(order)

    assert not [call for call in tr.calls if call[0] == "RequestData"]


def test_signal_style_callbacks_are_supported(tmp_path: Path):
    config = load_config(tmp_path / "missing.json")
    tr = SignalControl()
    real = SignalControl()

    Indi(tr, real, config)

    assert tr.ReceiveData.callback is not None
    assert tr.ReceiveSysMsg.callback is not None
    assert real.ReceiveRTData.callback is not None


def test_missing_callback_surface_fails_fast(tmp_path: Path):
    config = load_config(tmp_path / "missing.json")

    with pytest.raises(RuntimeError, match="callback"):
        Indi(NoCallbackControl(), NoCallbackControl(), config)
