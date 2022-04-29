import os
import numpy as np
from torch.utils.data import Dataset
import random


class RankReviewData(Dataset):

    def __init__(self, opt, mode):
        self.opt = opt
        if mode == 'Train':
            path = os.path.join(opt.root_path, 'train/')
            self.data = np.load(path + 'Train.npy', encoding='bytes')
            self.scores = np.load(path + 'Train_Score.npy')
        elif mode == 'Val':
            path = os.path.join(opt.root_path, 'val/')
            self.data = np.load(path + 'Val.npy', encoding='bytes')
            self.scores = np.load(path + 'Val_Score.npy')
        else:
            path = os.path.join(opt.root_path, 'test/')
            self.data = np.load(path + 'Test.npy', encoding='bytes')
            self.scores = np.load(path + 'Test_Score.npy')
        self.x = list(zip(self.data, self.scores))

        self.all_items = set()
        for idx in len(opt.user2itemid_list):
            self.all_items.union(set(opt.user2itemid_list[idx]))
            print(len(self.all_items), end='')
            if idx % 100 == 0: print()

    def __getitem__(self, idx):
        assert idx < len(self.x)

        user, pos_item = self.data[idx]
        # 负采样
        neg_item = random.sample(self.all_items.difference(self.opt.user2itemid_list[user]), 1)[0]

        return [user, pos_item, neg_item]

    def __len__(self):
        return len(self.x)
