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
random_tensor1 = torch.rand(size=(3,4)
                              ,dtype = torch.float16,
                              requires_grad=True)
random_tensor2 = torch.rand(size=(4,5)
                              ,dtype = torch.float16,
                              requires_grad=False)
result_tensor = torch.matmul(random_tensor1,random_tensor2)
print(f"result tensor: {result_tensor}")
print(f"result tensor shape: {result_tensor.shape}")
print(f"result tensor dimension: {result_tensor.ndim}")
print(f"result tensor datatype: {result_tensor.dtype}")

# transpose
ori_tensor = torch.rand(size = (3,4))
trans_tensor = ori_tensor.T
result_tensor = torch.matmul(ori_tensor,trans_tensor)
print(f"original tensor shape: {ori_tensor.shape}")
print(f"transposed tensor shape: {trans_tensor.shape}")
print(f"result tensor shape: {result_tensor.shape}")

# tensor aggregation
x = torch.arange(1,100,10)
y = torch.rand(size = (3,4))
# find mean
mean_x = torch.mean(x.type(torch.float32))
mean_y = torch.mean(y, axis = 0)
print(f"mean of x: {mean_x}")
print(f"mean of y: {mean_y}")
# find sum
sum_x = torch.sum(x)
sum_y = torch.sum(y, axis = 1)
print(f"sum of x: {sum_x}") 
print(f"sum of y: {sum_y}")
# find max
max_x = torch.max(x)
max_y = torch.max(y, axis = 1)
print(f"max of x: {max_x}")
print(f"max of y: {max_y}")

# find positional
argmax_x = torch.argmax(x)
argmax_y = torch.argmax(y, axis = 0)
print(f"argmax of x: {argmax_x}")
print(f"argmax of y: {argmax_y}")
print(f"argmax of y type: {argmax_y.shape}")

# Reshapin, stacking, squeezing, unsqueezing tensors
# reshape
x = torch.arange(1.,10.)
x_reshaped = x.reshape(3,3)
print(f"x reshaped: {x_reshaped}")

# stacking
x_stacked = torch.stack([x,x,x], axis=0)
print(f"x stacked: {x_stacked}")

#squeezing
x_squeeze = torch.squeeze(x_stacked, axis=0)
print(f"x squeezed: {x_squeeze}")

#unszueezing
x_unsqueeze = torch.unsqueeze(x_squeeze, axis=0)
print(f"x unsqueezed shape: {x_unsqueeze.shape}")

# permute
# permute is a view.
x_orirginal = torch.rand(size=(224,224,3))
x_permute = x_orirginal.permute(2,0,1)
print(f"x original shape: {x_orirginal.shape}")
print(f"x permuted shape: {x_permute.shape}")

#indexing and selecting data from tensors
x = torch.arange(1,10).reshape(1,3,3)
print(f"x: {x}")
print(f"x[0]: {x[0]}")
print(f"x[0,0]: {x[0][0]}")
print(f"x[0,0,0]: {x[0][0][0]}")

# use ":" to select all elements in a dimension
print(f"x[:,:,1]: {x[:,:,1]}")
print(f"x[:,1,1]: {x[:,1,1]}")

# Numpy to Torch tensor and vice versa
# numpy to torch tensor
array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array) # defalut dtype is float64
print(f"numpy array: {array}, type: {type(array)}")
print(f"torch tensor: {tensor}, type: {type(tensor)}")

# torch tensor to numpy
tensor = torch.ones(size = (3,4))
numpy_tensor = tensor.numpy() # default dtype is float32
print(f"torch tensor: {tensor}, type: {type(tensor)}")
print(f"numpy tensor: {numpy_tensor}, type: {type(numpy_tensor)}")
print(f"numpy tensor data type: {numpy_tensor.dtype}")