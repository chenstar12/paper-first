import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

'''
MSCI5：增加review list id embedding的利用率（获取fea）  ---- 几个实验证明，效果变差！！！！
'''


class MSCI7(nn.Module):

    def __init__(self, opt):
        super(MSCI7, self).__init__()
        self.opt = opt
        self.num_fea = 3  # 0,1,2 == id,doc,review

        self.user_net = Net(opt, 'user')
        self.item_net = Net(opt, 'item')

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, \
        user_doc, item_doc, user_sentiments, item_sentiments = datas

        u_fea = self.user_net(user_doc, user_reviews, uids, user_item2id, user_sentiments)  # Net的forward函数得：[128,2,32]
        i_fea = self.item_net(item_doc, item_reviews, iids, item_user2id, item_sentiments)  # [128,2,32]

        return u_fea, i_fea


class Net(nn.Module):
    def __init__(self, opt, uori):
        super(Net, self).__init__()
        self.opt = opt

        if uori == 'user':
            id_num = self.opt.user_num
            ui_id_num = self.opt.item_num
        else:  # item
            id_num = self.opt.item_num
            ui_id_num = self.opt.user_num

        self.id_embedding = nn.Embedding(id_num, self.opt.id_emb_size)  # user数/item数 * 32, 即：[几万，32]
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # 50000 * 300
        self.u_i_id_embedding = nn.Embedding(ui_id_num, self.opt.id_emb_size)  # embedding的搜索空间：[几万，32]

        self.cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))  # 卷积

        self.linear = nn.Linear(self.opt.filters_num + self.opt.id_emb_size,
                                self.opt.id_emb_size)  # [100,32].用来给review特征降维
        self.id_linear = nn.Linear(self.opt.id_emb_size, self.opt.id_emb_size, bias=False)  # [32,32]
        self.attention_linear = nn.Linear(self.opt.id_emb_size, 1)
        self.doc_linear = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)
        self.fc_layer = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)
        self.mix_layer = nn.Linear(self.opt.filters_num + self.opt.id_emb_size, self.opt.filters_num)

        self.dropout = nn.Dropout(self.opt.drop_out)
        self.reset_para()

    def forward(self, doc, reviews, ids, ids_list, sentiments):  # 添加了sentiments

        #  1. word embedding
        # reviews:用户[128, 10, 214] ->  [128, 10, 214, 300]，物品[128, 27, 214] ->  [128, 27, 214, 300]
        reviews = self.word_embs(reviews)
        bs, r_num, r_len, wd = reviews.size()
        reviews = reviews.view(-1, r_len, wd)  # [1280, 214, 300]
        # 2. cnn
        # 先unsqueeze(1) -> [1280,1,214,300]，再cnn -> [1280, 100, 212,1],最后squeeze(3) -> [1280, 100, 212]
        fea = F.relu(self.cnn(reviews.unsqueeze(1))).squeeze(3)
        fea = F.max_pool1d(fea, fea.size(2)).squeeze(2)  # [1280, 100]
        fea = fea.view(-1, r_num, fea.size(1))  # torch.Size([128, 10/27, 100])

        id_emb = self.id_embedding(ids)  # [128] -> [128, 32]
        u_i_id_emb = self.u_i_id_embedding(ids_list)  # [128,10/27] -> [128, 10/27, 32]

        #  3. attention（linear attention）
        #  rs_mix维度：user为[128,10,32]，item为[128,27，32]
        rs_mix = F.relu(  # 这一步的目的：把user(或item)的review特征表示和对应item(或user)ids embedding特征表示统一维度
            torch.cat([fea, F.relu(self.id_linear(u_i_id_emb))], dim=2)  # [128,10,132]
        )
        fea = F.relu(self.mix_layer(rs_mix))  # 降维 -> [128,10,100]

        rs_mix = self.linear(rs_mix)  # 用于计算注意力权重，[128,10,132] -> [128,10,32]
        att_score = self.attention_linear(rs_mix)  # 用全连接层实现 -> [128,10/27,1]，得到：某个user/item的每条review注意力权重
        att_weight = F.softmax(att_score, 1)  # 对第1维softmax，还是[128,10/27,1]

        r_fea = fea * att_weight  # fea:[128, 10/27, 100]; 得到r_fea也是[128, 10, 100]；原理：最后一维attention自动扩展100次
        # 矩阵的每个数都缩放了r_num倍；由于下面还要乘以weight，所以这里要乘r_num
        r_fea = r_fea * r_num

        '''
        （1）先把情感权重归一化 ---- softmax
        （2）乘以sentiment，subjectivity，vader的compound； 或者选其中一两个
        （3）上一步的特征相加除以2或3
        '''
        polarity_w = sentiments[:, :, 0]  # 获取第一列 ---- polarity
        polarity_w = polarity_w.unsqueeze(2)  # -> [128,10,1]
        polarity_w = polarity_w / 10000
        polarity_w = F.softmax(polarity_w, 1)
        # polarity_w把矩阵的每个数都缩放了r_num倍；由于下面还要乘以attention weight，所以这里要乘r_num
        r_fea = r_fea * polarity_w  # fea还是[128, 10/27, 100]

        r_fea = r_fea.sum(1)  # 每个user的10条特征(经过加权的特征)相加，相当于池化？ -> [128,100]

        r_fea = self.dropout(r_fea)

        '''
        添加了doc特征
        '''
        # 调用Embedding类的forward函数（F.embedding查找表）： torch.Size([50002, 300]) -> torch.Size([128, 500, 300])
        doc = self.word_embs(doc)  # [128, 500] -> [128, 500, 300]
        # unsqueeze(1): [128,500,300] -> [128,1,500,300]; cnn -> [128,100,498,1]; squeeze -> [128,100,498]
        doc_fea = F.relu(self.cnn(doc.unsqueeze(1))).squeeze(3)
        # 最大池化：[] -> [128,100，1] ，squeeze(2): -> [128,100],作为fc层的输入
        doc_fea = F.max_pool1d(doc_fea, doc_fea.size(2)).squeeze(2)
        doc_fea = F.relu(self.doc_linear(doc_fea))  # 降维 -> [128,32]

        # fc_layer:100*32,将r_fea：[128,100] -> [128,32]; 所以stack输入两个都是[128,32],输出[128,2,32]
        return torch.stack([F.relu(id_emb), doc_fea, F.relu(self.fc_layer(r_fea))], 1)  # 加入doc后 -> [128,3,32]

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

        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.constant_(self.linear.bias, 0.1)

        nn.init.uniform_(self.mix_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.mix_layer.bias, 0.1)

        nn.init.uniform_(self.doc_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.doc_linear.bias, 0.1)

        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)

        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)
