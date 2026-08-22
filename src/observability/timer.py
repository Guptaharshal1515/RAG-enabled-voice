import time
from typing import Optional


class Timer:
    """
    High-precision microsecond stopwatch for sub-stage latency profiling.
    """

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self) -> "Timer":
        self.start_time = time.perf_counter()
        self.end_time = None
        return self

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        return self.elapsed_ms()

    def elapsed_ms(self) -> float:
        if self.start_time is None:
            return 0.0
        current = self.end_time if self.end_time is not None else time.perf_counter()
        return round((current - self.start_time) * 1000, 3)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
