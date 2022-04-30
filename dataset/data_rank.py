import os
import numpy as np
from torch.utils.data import Dataset
import random

'''
正样本：rating >= 4 或 rating >= 3.5
负采样：在剩余样本中random sampling 、（备选方案）优先选择已交互的低rating样本

注 ---- 参考MACR：有交互/评分为1；为交互为0；！！！！！！
'''


class RankReviewData(Dataset):

    def __init__(self, opt, mode):
        self.opt = opt
        if mode == 'Train':
            path = os.path.join(opt.data_root, 'train/')
            self.data = np.load(path + 'Train.npy', encoding='bytes')
            self.scores = np.load(path + 'Train_Score.npy')
        elif mode == 'Val':
            path = os.path.join(opt.data_root, 'val/')
            self.data = np.load(path + 'Val.npy', encoding='bytes')
            self.scores = np.load(path + 'Val_Score.npy')
        else:
            path = os.path.join(opt.data_root, 'test/')
            self.data = np.load(path + 'Test.npy', encoding='bytes')
            self.scores = np.load(path + 'Test_Score.npy')
        self.x = list(zip(self.data, self.scores))

        self.all_items = set(range(len(opt.item2userid_list)))
        print('len(self.all_items): ',len(self.all_items))

    def __getitem__(self, idx):
        assert idx < len(self.x)

        user, pos_item = self.data[idx]
        # 负采样
        neg_item = random.sample(self.all_items.difference(self.opt.user2itemid_list[user]), 1)[0]

        return [user, pos_item, neg_item]

    def __len__(self):
        return len(self.x)
