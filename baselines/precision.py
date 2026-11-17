"""Scoped IEEE FP32 execution for cancellation-sensitive native NCC losses."""
from contextlib import contextmanager
import torch


@contextmanager
def ieee_fp32():
    convolution = torch.backends.cudnn.allow_tf32
    multiplication = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = convolution
        torch.backends.cuda.matmul.allow_tf32 = multiplication
