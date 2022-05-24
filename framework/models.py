import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .prediction import PredictionLayer
from .fusion import FusionLayer


class Model(nn.Module):

    def __init__(self, opt, Net):
        super(Model, self).__init__()
        self.opt = opt
        self.model_name = self.opt.model
        self.net = Net(opt)

        if self.opt.ui_merge == 'cat':
            if self.opt.r_id_merge == 'cat':
                if opt.model in ['MSCI0F', 'MSCI0Y1']:
                    feature_dim = self.opt.id_emb_size * self.opt.num_fea * 4
                else:
                    feature_dim = self.opt.id_emb_size * self.opt.num_fea * 2
            else:
                feature_dim = self.opt.id_emb_size * 2
        else:
            if self.opt.r_id_merge == 'cat':
                feature_dim = self.opt.id_emb_size * self.opt.num_fea
            else:
                feature_dim = self.opt.id_emb_size

        self.opt.feature_dim = feature_dim
        self.fusion_net = FusionLayer(opt)  # fusion层！！！
        self.predict_net = PredictionLayer(opt)  # predict层！！！
        self.dropout = nn.Dropout(self.opt.drop_out)

    def forward(self, datas, opt):
        if opt.stage == 'train':
            user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, \
            user_doc, item_doc, user_sentiments, item_sentiments, ui_senti = datas
        else:
            user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, \
            user_doc, item_doc, user_sentiments, item_sentiments = datas

        user_feature, item_feature = self.net(datas)  # 如：DeepConn输出的u_fea,i_fea
        # print(item_feature[:, 1, :].shape)

        # opt.pos_u.extend(np.array(user_feature[opt.pos_idx, 1, :]).tolist())
        # opt.pos_i.extend(np.array(item_feature[opt.pos_idx, 1, :]).tolist())
        #
        # opt.neg_u.extend(np.array(user_feature[opt.neg_idx, 1, :]).tolist())
        # opt.neg_i.extend(np.array(item_feature[opt.neg_idx, 1, :]).tolist())
        if opt.stage == 'test':
            ifea = item_feature[:, 1, :]
            opt.ifea.extend(ifea.numpy().tolist())
            print(len(opt.ifea))
            print(len(opt.ifea[0]))
            print(len(opt.ifea[1]))

        # fusion feature,如DeepCoNN的cat得到[128,64]
        ui_feature = self.fusion_net(user_feature, item_feature)  # NARRE是[128,64]
        ui_feature = self.dropout(ui_feature)  # 还是[128,64]
        output = self.predict_net(ui_feature, uids, iids).squeeze(1)  # pred:[128]

        polarity = user_sentiments[:, :, 0]  # 获取第1列 [128,10]
        subjectivity = user_sentiments[:, :, 1]  # 获取第2列 [128,10]
        polarity_i = item_sentiments[:, :, 0]  # 获取第1列
        num = polarity.shape[1]
        num_i = polarity_i.shape[1]
        polarity = polarity_i.sum(dim=1) / (10000 * num_i)  # item的平均分（也可用score的均值）
        subjectivity = subjectivity.sum(dim=1) / (10000 * num)  # user的主观性

        if opt.stage == 'train':

            if opt.inference == '':
                return output
            elif opt.inference[:5] == 'trans':  # 正确的调参
                # po = ui_senti[:, 0] / 10000  # 1e4装个逼
                # sub = ui_senti[:, 1] / 10000
                # c = ui_senti[:, 2] / 10000

                if self.opt.inference in ['trans-tanh']:
                    output = output + output * self.opt.lambda1 * torch.tanh(polarity * subjectivity)
                if self.opt.inference in ['trans-PD1']:
                    pass
                    # output = output + output * self.opt.lambda1 * torch.sigmoid(po * sub)
                if self.opt.inference in ['trans-PDA']:  # 调参：lambda2
                    pass
                    # tmp = po ** self.opt.lambda2
                    # df = pd.DataFrame(tmp.cpu())
                    # df.fillna(df.mean(), inplace=True)  # 均值填充
                    # tmp = torch.from_numpy(df.values).squeeze(1).cuda()
                    # output = output * torch.tanh(tmp)  # 新增激活函数----sigmoid

                return output

        else:
            if self.opt.ei == '':  # eval时的inference
                return output
            elif self.opt.inference in ['trans-tanh']:
                output = output + output * self.opt.lambda1 * torch.tanh(polarity * subjectivity)
                return output
            else:
                output = output + output * self.opt.lambda1 * torch.sigmoid(polarity * subjectivity)
                return output

    def load(self, path):
        '''
        加载指定模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, epoch=None, name=None, opt=None):
        '''
        保存模型
        '''
        prefix = '/content/drive/MyDrive/checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + self.model_name + '_' + str(name) + '_' + str(opt) + '.pth'
        torch.save(self.state_dict(), name)
        return name
