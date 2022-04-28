import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class MSCI0D1T4(nn.Module):
    '''
    BN
    '''

    def __init__(self, opt):
        super(MSCI0D1T4, self).__init__()
        self.opt = opt
        self.num_fea = 2  # 0,1,2 == id,doc,review

        self.user_net = Net(opt, 'user')
        self.item_net = Net(opt, 'item')

    def forward(self, datas):
        if self.opt.stage == 'train':
            user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, \
            user_doc, item_doc, user_sentiments, item_sentiments, _ = datas
        else:
            user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, \
            user_doc, item_doc, user_sentiments, item_sentiments = datas

        u_fea = self.user_net(user_reviews, uids, user_item2id, user_sentiments)  # 下面Net的forward函数得：[128,2,32]
        i_fea = self.item_net(item_reviews, iids, item_user2id, item_sentiments)  # [128,2,32]

        return u_fea, i_fea


class Net(nn.Module):
    def __init__(self, opt, uori):
        super(Net, self).__init__()
        self.opt = opt

        if uori == 'user':
            id_num = self.opt.user_num
        else:  # item
            id_num = self.opt.item_num

        self.id_embedding = nn.Embedding(id_num, self.opt.id_emb_size)  # user数/item数 * 32, 即：[几万，32]
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # 50000 * 300
        self.cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))  # 卷积
        self.fc_layer = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)
        self.dropout = nn.Dropout(self.opt.drop_out)
        self.reset_para()

    def forward(self, reviews, ids, ids_list, sentiments):  # 添加了sentiments
        #  1. word embedding
        # reviews:用户[128, 10, 214] ->  [128, 10, 214, 300]，物品[128, 27, 214] ->  [128, 27, 214, 300]
        reviews = self.word_embs(reviews)
        bs, r_num, r_len, wd = reviews.size()
        reviews = reviews.view(-1, r_len, wd)  # [1280, 214, 300]

        # 2. cnn
        # 先unsqueeze(1) -> [1280,1,214,300]，再cnn -> [1280, 100, 212,1],最后squeeze(3) -> [1280, 100, 212]
        fea = F.relu(self.cnn(reviews.unsqueeze(1))).squeeze(3)
        fea = F.max_pool1d(fea, fea.size(2)).squeeze(2)  # [1280, 100]
        # bn1 = nn.BatchNorm1d(self.opt.filters_num, affine=True).cuda()
        # fea = bn1(fea)
        fea = fea.view(-1, r_num, fea.size(1))  # torch.Size([128, 10/27, 100])

        bn2 = nn.BatchNorm1d(r_num, affine=True).cuda()
        r_fea=bn2(fea)

        id_emb = self.id_embedding(ids)  # [128] -> [128, 32]

        '''
        （1）先把情感权重归一化 ---- softmax
        （2）乘以sentiment，subjectivity，vader的compound； 或者选其中一两个
        （3）上一步的特征相加除以2或3
        '''
        polarity_w = sentiments[:, :, 0]  # 获取第1列 ---- polarity
        polarity_w = polarity_w.unsqueeze(2)  # -> [128,10,1]
        polarity_w = polarity_w / 10000
        polarity_w = F.softmax(polarity_w, 1)

        subj_w = sentiments[:, :, 1]  # 获取第2列 ---- subj
        subj_w = subj_w.unsqueeze(2)  # -> [128,10,1]
        subj_w = subj_w / 10000
        subj_w = F.softmax(subj_w, 1)

        r_fea = fea
        r_fea = r_fea * polarity_w

        # r_fea = self.dropout(r_fea)
        r_fea = r_fea * r_num

        r_fea = r_fea * subj_w
        # bn3 = nn.BatchNorm1d(r_num, affine=True).cuda()
        # r_fea=bn3(r_fea)
        r_fea = r_fea * r_num
        # r_fea = self.dropout(r_fea)

        r_fea = r_fea.sum(1)  # 每个user的10条特征(经过加权的特征)相加，相当于池化？ -> [128,100]
        r_fea = self.dropout(r_fea)
        # fc_layer:100*32,将r_fea：[128,100] -> [128,32]; 所以stack输入两个都是[128,32],输出[128,2,32]
        return torch.stack([F.relu(id_emb), F.relu(self.fc_layer(r_fea))], 1)

    def reset_para(self):
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())  # word_embs：直接复制w2v矩阵
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)

        nn.init.xavier_normal_(self.id_embedding.weight)

        nn.init.xavier_normal_(self.cnn.weight)
        nn.init.constant_(self.cnn.bias, 0.1)

        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)
