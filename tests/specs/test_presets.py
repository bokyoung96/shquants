from backtesting.specs import get_preset


def test_revision_signal_preset_uses_signal_dates_schedule() -> None:
    spec = get_preset("kospi200_revision_signal")

    assert spec.strategy == "revision_signal"
    assert spec.name == "kospi200_revision_signal_close_v1"
    assert spec.schedule.kind == "signal_dates"
    assert spec.schedule.name is None
    assert spec.fill_mode == "close"
    assert spec.use_k200 is True
    assert spec.top_n == 0
    assert spec.lookback == 20
    assert spec.warmup_days == 180
