# utils/logger_decorator.py
import functools
import time
from typing import Callable

def structured_logger(name: str):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(state, *args, **kwargs):
            start_time = time.time()
            print(f"🌀 [{name}] Inicio - {state.get('input', '')[:40]}...")
            result = func(state, *args, **kwargs)
            duration = round(time.time() - start_time, 3)
            print(f"✅ [{name}] Fin ({duration}s)")
            return result
        return wrapper
    return decorator