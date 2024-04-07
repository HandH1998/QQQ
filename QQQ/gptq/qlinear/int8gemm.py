
import torch
import threading
from QQQ._CUDA import I8CUGEMM

class Int8GEMM(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        if not hasattr(self, "i8cugemm"):
            self.i8cugemm = I8CUGEMM()

    def __new__(cls, *args, **kwargs):
        if not hasattr(Int8GEMM, "_instance"):
            with Int8GEMM._instance_lock:
                if not hasattr(Int8GEMM, "_instance"):
                    Int8GEMM._instance = object.__new__(cls)
        return Int8GEMM._instance

    def get_i8cugemm(self):
        return self.i8cugemm