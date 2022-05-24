import torch

t1 = torch.tensor([2, 5, 7, 1])
t1, idx = torch.sort(t1)
print(t1,idx)
