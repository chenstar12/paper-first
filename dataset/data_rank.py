import os
import numpy as np
from torch.utils.data import Dataset


class RankReviewData(Dataset):

    def __init__(self, root_path, mode):
        if mode == 'Train':
            path = os.path.join(root_path, 'train/')
            self.data = np.load(path + 'Train.npy', encoding='bytes')
            self.scores = np.load(path + 'Train_Score.npy')
        elif mode == 'Val':
            path = os.path.join(root_path, 'val/')
            self.data = np.load(path + 'Val.npy', encoding='bytes')
            self.scores = np.load(path + 'Val_Score.npy')
        else:
            path = os.path.join(root_path, 'test/')
            self.data = np.load(path + 'Test.npy', encoding='bytes')
            self.scores = np.load(path + 'Test_Score.npy')
        self.x = list(zip(self.data, self.scores))

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)
