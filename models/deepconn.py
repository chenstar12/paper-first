import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DeepCoNN(nn.Module):

    def __init__(self, opt, uori='user'):
        super(DeepCoNN, self).__init__()
        self.opt = opt
        self.num_fea = 1  # DOC

        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # torch.Size([50002, 300])
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # torch.Size([50002, 300])
        # 输出通道数 ---- opt.filters_num ---- 100； 卷积核大小 ---- (3,300)
        self.user_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        self.item_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))

        self.user_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)
        self.item_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)  # 100,32
        self.dropout = nn.Dropout(self.opt.drop_out)

        self.reset_para()  # 模型参数 ---- 初始化！！！

    def forward(self, datas):  # 依次调用各nn.Module子类的forward函数
        _, _, uids, iids, _, _, user_doc, item_doc = datas  # user_doc形状：torch.Size([128, 500])

        # 调用Embedding类的forward函数（F.embedding查找表）： torch.Size([50002, 300]) -> torch.Size([128, 500, 300])
        user_doc = self.user_word_embs(user_doc)  # torch.Size([128, 500, 300])
        item_doc = self.item_word_embs(item_doc)  # torch.Size([128, 500, 300])
        # unsqueeze(1): [128,500,300] -> [128,1,500,300]; cnn+squeeze: [] -> [128,100,498]
        u_fea = F.relu(self.user_cnn(user_doc.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        i_fea = F.relu(self.item_cnn(item_doc.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        # 最大池化：[] -> [128,100，1] ，squeeze(2): -> [128,100],作为fc层的输入
        u_fea = F.max_pool1d(u_fea, u_fea.size(2)).squeeze(2)
        i_fea = F.max_pool1d(i_fea, i_fea.size(2)).squeeze(2)
        # fc层：[128,100] -> [128,32]
        u_fea = self.dropout(self.user_fc_linear(u_fea))
        i_fea = self.dropout(self.item_fc_linear(i_fea))
        
        return torch.stack([u_fea], 1), torch.stack([i_fea], 1)  # torch.Size([128, 1, 32])

    def reset_para(self):

        for cnn in [self.user_cnn, self.item_cnn]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for fc in [self.user_fc_linear, self.item_fc_linear]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))  # w2v: torch.Size([50002, 300])
            if self.opt.use_gpu:
                self.user_word_embs.weight.data.copy_(w2v.cuda())
                self.item_word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.user_word_embs.weight.data.copy_(w2v)
                self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)
