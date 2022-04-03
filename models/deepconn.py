import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DeepCoNN(nn.Module):

    def __init__(self, opt, uori='user'):
        super(DeepCoNN, self).__init__()
        self.opt = opt
        self.num_fea = 1  # DOC

        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        print('================================embedding========================================')
        print('user_word_embs', self.user_word_embs.shape)
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        print('================================embedding========================================')
        print('item_word_embs', self.item_word_embs.shape)
        # 输出通道数 ---- opt.filters_num ---- 100； 卷积核大小 ---- (3,300)
        self.user_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        self.item_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))

        self.user_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)
        self.item_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)  # 100,32
        self.dropout = nn.Dropout(self.opt.drop_out)

        self.reset_para()  # 模型参数 ---- 初始化！！！

    def forward(self, datas):
        _, _, uids, iids, _, _, user_doc, item_doc = datas

        user_doc = self.user_word_embs(user_doc)
        print('=============================================user_doc.shape============================')
        print(user_doc.shape)
        item_doc = self.item_word_embs(item_doc)
        print('=============================================item_doc.shape============================')
        print(item_doc.shape)

        u_fea = F.relu(self.user_cnn(user_doc.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        print('u_fea.shape: ', u_fea.shape)
        i_fea = F.relu(self.item_cnn(item_doc.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        print('i_fea.shape: ', i_fea.shape)

        u_fea = F.max_pool1d(u_fea, u_fea.size(2)).squeeze(2)
        print('maxpooled u_fea.shape: ', u_fea.shape)
        i_fea = F.max_pool1d(i_fea, i_fea.size(2)).squeeze(2)
        print('maxpooled i_fea.shape: ', i_fea.shape)

        u_fea = self.dropout(self.user_fc_linear(u_fea))
        print('fc  u_fea.shape: ', u_fea.shape)

        i_fea = self.dropout(self.item_fc_linear(i_fea))
        print('fc i_fea.shape: ', i_fea.shape)
        print('stack u:', torch.stack([u_fea], 1).shape)
        print('stack i:', torch.stack([i_fea], 1).shape)
        return torch.stack([u_fea], 1), torch.stack([i_fea], 1)

    def reset_para(self):

        for cnn in [self.user_cnn, self.item_cnn]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for fc in [self.user_fc_linear, self.item_fc_linear]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.user_word_embs.weight.data.copy_(w2v.cuda())
                self.item_word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.user_word_embs.weight.data.copy_(w2v)
                self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)
