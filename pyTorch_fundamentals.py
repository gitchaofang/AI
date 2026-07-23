import torch
from torch import nn
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)

#basics
ts_np = torch.tensor(
    np.array([[1, 1], [2, 2]]),
    dtype=torch.float32,
    requires_grad=True
)
print(f"Tensor dimension: {ts_np.ndim}")
print(f"shape of tensor:{ts_np.shape}")
print(f"first element of tensor: {ts_np[0,1]}")
out = ts_np.pow(2).sum()   # scalar
out.backward()

print("loss:", out.item())
print("gradient:", ts_np.grad)

#random tensors
random_ts = torch.rand(3,4)
print(f"random tensor:{random_ts}")
print(f"random tensor shape: {random_ts.shape}")
print(f"random tensor dimension: {random_ts.ndim}")
random_image_ts = torch.rand(size=(224,224,3), dtype=torch.float32)
print(f"random image tensor shape: {random_image_ts.shape}")

#zeros and ones tensors
zero_ts = torch.zeros(size=(6,))
print(f"zero tensor shape: {zero_ts.shape}")
print(f"zero tensor dimension: {zero_ts.ndim}")

one_ts = torch.ones(1)
print(f"one tensor shape: {one_ts.shape}")
print(f"one tensor dimension: {one_ts.ndim}")
print(f"one tensort data type: {one_ts.dtype}")

# a range of tensors
arange_ts = torch.arange(start=0,end=20,step=2)
print(f"arange tensor: {arange_ts}")
print(f"arange tensor shape: {arange_ts.shape}")
print(f"arange tensor dimension: {arange_ts.ndim}")

# tensor like
ten_zeros = torch.zeros_like(input=arange_ts)
print(f"tensor like shape: {ten_zeros.shape}")

# tensor datatypes
float32_tensor = torch.tensor([[3,4],[1,5.0]]
                              ,dtype = None
                              ,requires_grad=True
                              ,device=None)  # Three most important arguments of torch.tensor are dtype, requires_grad, and device.
print(f"float32 tensor data type: {float32_tensor.dtype}")

# getting info from tensors
some_tensor = torch.rand(size=(3,4))
print(f"some tensor: {some_tensor}")
print(f"some tensor shape: {some_tensor.shape}")
print(f"some tensor dimension: {some_tensor.ndim}")
print(f"some tensor datatype: {some_tensor.dtype}")

# tensor operations
# addition, subtraction, multiplication(element-wise), division, matrix multiplication

tensor = torch.tensor([1,2,3])
tensor_plus = tensor + 10
print(f"tensor plus 10: {tensor_plus}")
tensor_ele_mul = tensor * tensor # * = torch.mul() for element-wise multiplication
print(f"tensor element-wise multiplication: {tensor_ele_mul}")
tensor_mat_mul = torch.matmul(tensor,tensor) #tensor @ tensor also works for matrix multiplication
print(f"tensor matrix multiplication: {tensor_mat_mul}")

