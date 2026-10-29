import time
from contextlib import contextmanager

class Timer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.elapsed = 0.0
        self.running = False
    
    def start(self):
        if not self.running:
            self.start_time = time.perf_counter()
            self.running = True
    
    def stop(self):
        if self.running:
            self.elapsed += time.perf_counter() - self.start_time
            self.running = False
        return self.elapsed
    
    def get_elapsed_ms(self):
        if self.running:
            return (self.elapsed + time.perf_counter() - self.start_time) * 1000
        return self.elapsed * 1000

@contextmanager
def timer_context():
    t = Timer()
    t.start()
    yield t
    t.stop()
