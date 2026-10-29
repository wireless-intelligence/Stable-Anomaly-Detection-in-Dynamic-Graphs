import psutil
import torch

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_mem_gb = mem_info.rss / (1024 ** 3)
    
    gpu_mem_gb = 0.0
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    return {
        'cpu_mem_gb': cpu_mem_gb,
        'gpu_mem_gb': gpu_mem_gb,
        'total_mem_gb': cpu_mem_gb + gpu_mem_gb
    }

def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

class MemoryTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        reset_memory_stats()
        self.peak_cpu_mem = 0.0
        self.peak_gpu_mem = 0.0
    
    def update(self):
        mem = get_memory_usage()
        self.peak_cpu_mem = max(self.peak_cpu_mem, mem['cpu_mem_gb'])
        self.peak_gpu_mem = max(self.peak_gpu_mem, mem['gpu_mem_gb'])
    
    def get_peak(self):
        return {
            'peak_cpu_mem_gb': self.peak_cpu_mem,
            'peak_gpu_mem_gb': self.peak_gpu_mem,
            'peak_total_mem_gb': self.peak_cpu_mem + self.peak_gpu_mem
        }
