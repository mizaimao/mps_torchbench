"""Prints available devices for torch."""

import torch

print("MPS found:", torch.backends.mps.is_available())
print("CUDA found:", torch.cuda.is_available())