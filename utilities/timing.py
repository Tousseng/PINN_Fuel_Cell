import time, datetime
from typing import Callable

def timer(func: Callable) -> Callable:
    """

    Args:
        func: Function that wants to be timed.

    Returns:
        The result of the function evaluation in the first place
        and the duration in the second place of the returned tuple.
    """
    def wrapper(*args, **kwargs) -> tuple[any,float]:
        timer_start: float = time.time()
        result = func(*args, **kwargs)
        duration: float = time.time() - timer_start
        return result, duration
    return wrapper

def convert_time(time_s: float) -> str:
    hours: int = int(time_s / 3600)
    time_s = time_s - hours * 3600
    minutes: int = int(time_s / 60)
    time_s = time_s - minutes * 60
    seconds: float = time_s
    return "{h:02d}:{m:02d}:{s:2.4f} (h:m:s)".format(h=hours, m=minutes, s=seconds)

def get_datetime() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")