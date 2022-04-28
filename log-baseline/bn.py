import torch.nn as nn
import torch

x = torch.zeros([4,3],requires_grad=True)
print(x)

nn.init.uniform_(x)
print(x)

bn=nn.BatchNorm1d(3,affine=True)
x=bn(x)
print(x)