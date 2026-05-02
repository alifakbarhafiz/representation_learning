"""Timing helpers using wall-clock time."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


class _ElapsedProxy:
    """A small proxy so `t.elapsed` and `t.elapsed()` both work."""

    def __init__(self, timer: "Timer"):
        self._timer = timer

    def __call__(self) -> float:
        return self._timer._elapsed_seconds()

    def __float__(self) -> float:  # pragma: no cover
        return self.__call__()


@dataclass
class Timer:
    """Simple wall-clock timer.

    Usage:
        t = Timer().start()
        ...
        t.stop()
        seconds = t.elapsed()

    Or:
        with Timer() as t:
            ...
        seconds = t.elapsed
    """

    _start: Optional[float] = None
    _end: Optional[float] = None

    def start(self) -> "Timer":
        self._start = time.time()
        self._end = None
        return self

    def stop(self) -> "Timer":
        if self._start is None:
            raise RuntimeError("Timer.stop() called before Timer.start().")
        self._end = time.time()
        return self

    def _elapsed_seconds(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer elapsed called before start().")
        end = self._end if self._end is not None else time.time()
        return float(end - self._start)

    @property
    def elapsed(self) -> _ElapsedProxy:
        """Elapsed time proxy.

        - `t.elapsed` returns a proxy that can be read (float-cast) or called.
        - `t.elapsed()` returns seconds.
        """

        return _ElapsedProxy(self)

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.stop()

    def elapsed_seconds(self) -> float:
        """Elapsed seconds after context manager completes."""
        return self._elapsed_seconds()


def log_time(label: str, seconds: float) -> None:
    """Print a nicely formatted timing message."""

    print(f"[⏱ Timer] {label}: {seconds:.2f}s")


if __name__ == "__main__":
    with Timer() as t:
        time.sleep(0.1)
    log_time("Timer sanity", t.elapsed_seconds)

