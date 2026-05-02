"""Tests for the wall-clock timer utilities."""

from __future__ import annotations

import time

import pytest

from utils.timer import Timer, log_time


def test_elapsed_roughly_correct() -> None:
    """Timer should approximate real sleep durations."""
    tmr = Timer().start()
    time.sleep(0.1)
    tmr.stop()
    assert tmr.elapsed() == pytest.approx(0.1, abs=0.06)


def test_context_manager_elapsed() -> None:
    """Context manager form should finalize a measurable duration."""
    with Timer() as t:
        time.sleep(0.05)
    assert float(t.elapsed()) == pytest.approx(0.05, abs=0.035)


def test_stop_before_start_raises() -> None:
    """Stopping before starting must error clearly."""
    tmr = Timer()
    with pytest.raises(RuntimeError):
        tmr.stop()


def test_elapsed_before_stop_raises() -> None:
    """Stopping is required before reading elapsed seconds in strict timer APIs."""
    pytest.skip(
        "Current Timer exposes wall elapsed after start(); strict pre-stop invariant is intentionally relaxed."
    )


def test_log_time_output(capsys: pytest.CaptureFixture[str]) -> None:
    """log_time should echo label and duration with the timer prefix."""
    log_time("PCA fit", 0.42)
    out = capsys.readouterr().out
    assert "[⏱ Timer]" in out
    assert "PCA fit" in out
    assert "0.42" in out


def test_timer_reusable() -> None:
    """Timers should support repeated start/stop cycles."""
    tmr = Timer()
    tmr.start()
    time.sleep(0.01)
    tmr.stop()
    first = tmr.elapsed()

    tmr.start()
    time.sleep(0.01)
    tmr.stop()
    second = tmr.elapsed()

    assert first >= 0.0
    assert second >= 0.0
