import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .prediction import PredictionLayer
from .fusion import FusionLayer


class ModelVader(nn.Module):

    def __init__(self, opt, Net):
        super(ModelVader, self).__init__()
        self.opt = opt
        self.model_name = self.opt.model
        self.net = Net(opt)

        if self.opt.ui_merge == 'cat':
            if self.opt.r_id_merge == 'cat':
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
            user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc, user_sentiments, item_sentiments = datas

        user_feature, item_feature = self.net(datas)  # 如：DeepConn输出的u_fea,i_fea

        # fusion feature,如DeepCoNN的cat得到[128,64]
        ui_feature = self.fusion_net(user_feature, item_feature)  # NARRE是[128,64]
        ui_feature = self.dropout(ui_feature)  # 还是[128,64]
        output = self.predict_net(ui_feature, uids, iids).squeeze(1)  # pred:[128]

        if opt.stage == 'train':
            c = ui_senti[:, 2] / 10000
            if self.opt.inference in ['trans-PD']:
                output = output + output * self.opt.lambda1 * torch.tanh(c)  # T4
            if self.opt.inference in ['trans-PDA']:  # 调参：lambda2
                tmp = c ** self.opt.lambda2
                df = pd.DataFrame(tmp.cpu())
                df.fillna(df.mean(), inplace=True)  # 均值填充
                tmp = torch.from_numpy(df.values).squeeze(1).cuda()
                output = output * torch.tanh(tmp)  # 新增激活函数----sigmoid
            return output
        else:
            u_c = user_doc[:, -1]
            i_c = item_doc[:, -1]

            output = output + output * self.opt.lambda1 * torch.tanh(i_c)
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
        prefix = 'checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + self.model_name + '_' + str(name) + '_' + str(opt) + '.pth'
        torch.save(self.state_dict(), name)
        return name
