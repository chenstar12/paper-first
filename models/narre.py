import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NARRE(nn.Module):
    '''
    NARRE: WWW 2018
    '''

    def __init__(self, opt):
        super(NARRE, self).__init__()
        self.opt = opt
        self.num_fea = 2  # ID + DOC + Review

        self.user_net = Net(opt, 'user')
        self.item_net = Net(opt, 'item')

    def forward(self, datas):
        if self.opt.stage == 'train':
            user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, \
            user_doc, item_doc, user_sentiments, item_sentiments, _ = datas
        else:
            user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, \
            user_doc, item_doc, user_sentiments, item_sentiments = datas
        u_fea = self.user_net(user_reviews, uids, user_item2id)  # 有下面Net的forward函数得：[128,2,32]
        i_fea = self.item_net(item_reviews, iids, item_user2id)  # [128,2,32]
        return u_fea, i_fea


class Net(nn.Module):
    def __init__(self, opt, uori='user'):
        super(Net, self).__init__()
        self.opt = opt

        if uori == 'user':
            id_num = self.opt.user_num
            ui_id_num = self.opt.item_num
        else:
            id_num = self.opt.item_num
            ui_id_num = self.opt.user_num

        self.id_embedding = nn.Embedding(id_num, self.opt.id_emb_size)  # user/item num * 32,即：[几万，32]
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300
        self.u_i_id_embedding = nn.Embedding(ui_id_num, self.opt.id_emb_size)  # embedding的搜索空间：[几万，32]

        self.cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))  # 卷积

        self.review_linear = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)  # [100,32]
        self.id_linear = nn.Linear(self.opt.id_emb_size, self.opt.id_emb_size, bias=False)  # [32,32]
        self.attention_linear = nn.Linear(self.opt.id_emb_size, 1)
        self.fc_layer = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)

        self.dropout = nn.Dropout(self.opt.drop_out)
        self.reset_para()

    def forward(self, reviews, ids, ids_list):
        #  1. word embedding
        # reviews:[128, 10, 214] ->  [128, 10, 214, 300],其中：u_max_r = 10（物品：i_max_r=27）, r_max_len=214
        reviews = self.word_embs(reviews)
        bs, r_num, r_len, wd = reviews.size()
        reviews = reviews.view(-1, r_len, wd)  # [1280, 214, 300]

        # 2. cnn for review
        # 先unsqueeze(1) -> [1280,1,214,300]，再cnn -> [1280, 100, 212,1],最后squeeze(3) -> [1280, 100, 212]
        fea = F.relu(self.cnn(reviews.unsqueeze(1))).squeeze(3)
        fea = F.max_pool1d(fea, fea.size(2)).squeeze(2)  # [1280, 100]
        fea = fea.view(-1, r_num, fea.size(1))  # torch.Size([128, 10/27, 100])

        id_emb = self.id_embedding(ids)  # [128] -> [128, 32]
        u_i_id_emb = self.u_i_id_embedding(ids_list)  # [128,10/27] -> [128, 10/27, 32]

        #  3. attention（linear attention） ——> rs_mix维度：user为[128,10,32]，item为[128,27，32]
        rs_mix = F.relu(  # 这一步的目的：把user(或item)的review特征表示和对应item(或user)ids embedding特征表示统一维度
            self.review_linear(fea) +  # review降维:[128,10/27,100]->[128,10/27,32]
            self.id_linear(F.relu(u_i_id_emb))  # id降维后还是[128,10/27，32]
        )

        att_score = self.attention_linear(rs_mix)  # 用全连接层实现 -> [128,10/27,1]，得到：某个user/item的每条review注意力权重
        att_weight = F.softmax(att_score, 1)  # 还是[128,10/27,1]

        r_fea = fea * att_weight  # fea:[128, 10, 100]; 得到的r_fea也是[128, 10, 100]；原理：broadcast（最后一维的attention自动扩展到100个）
        r_fea = r_fea.sum(1)  # 每个user的10条特征相加，相当于池化？ -> [128,100]
        r_fea = self.dropout(r_fea)
        # fc_layer:100*32,将r_fea：[128,100] -> [128,32]; 所以stack输入两个都是[128,32],输出[128,2,32]
        return torch.stack([id_emb, self.fc_layer(r_fea)], 1)

    def reset_para(self):
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())  # word_embs：直接复制w2v矩阵
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)

        nn.init.uniform_(self.id_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.u_i_id_embedding.weight, a=-0.1, b=0.1)

        nn.init.xavier_normal_(self.cnn.weight)
        nn.init.constant_(self.cnn.bias, 0.1)

        nn.init.uniform_(self.id_linear.weight, -0.1, 0.1)

        nn.init.uniform_(self.review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.review_linear.bias, 0.1)

        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)

        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)
