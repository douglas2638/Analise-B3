# app/core/rate_limiting.py
import time
import random
from functools import wraps

class RateLimiter:
    def __init__(self, max_calls: int = 10, period: int = 60):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove calls outside the current period
            self.calls = [call for call in self.calls if now - call < self.period]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time + random.uniform(0.1, 0.5))
                    self.calls = self.calls[1:]
            
            self.calls.append(now)
            return func(*args, **kwargs)
        return wrapper

# Instância global
yahoo_rate_limiter = RateLimiter(max_calls=8, period=60)  # 8 requests por minuto