import torch
import torch.nn as nn

x = torch.empty(3, 5)
x = nn.init.normal_(x)
print(x)

y = x.unsqueeze(1)
print(y)
print(y.shape)
x = torch.stack([x], 1)
print(x)
print(x.shape)
