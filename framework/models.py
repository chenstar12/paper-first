import numpy as np
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
        if self.opt.model[:4] == 'MSCI':  # 获取所有数据(添加sentiment数据)
            user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, \
            user_doc, item_doc, user_sentiments, item_sentiments = datas
        else:
            user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, \
            user_doc, item_doc = datas

        user_feature, item_feature = self.net(datas)  # 如：DeepConn输出的u_fea,i_fea

        # fusion feature,如DeepCoNN的cat得到[128,64]
        ui_feature = self.fusion_net(user_feature, item_feature)  # NARRE是[128,64]
        ui_feature = self.dropout(ui_feature)  # 还是[128,64]
        output = self.predict_net(ui_feature, uids, iids).squeeze(1)  # pred:[128]

        print(opt.stage)
        if opt.stage == 'train':
            return output
        else:
            '''
            调参：lambda1和lambda2
            '''
            polarity = user_sentiments[:, :, 0]  # 获取第1列
            subjectivity = user_sentiments[:, :, 1]  # 获取第2列
            num = polarity.shape[1]

            polarity = polarity.sum(dim=1) / (10000 * num)
            subjectivity = subjectivity.sum(dim=1) / (10000 * num)

            if self.opt.inference in ['PD']:
                output = output + output * self.opt.lambda1 * polarity
            if self.opt.inference in ['PD1']:
                output = output + output * self.opt.lambda1 * polarity * subjectivity
                # print(polarity - subjectivity)
            elif self.opt.inference in ['PDA']:  # 调参：lambda2
                tmp = polarity ** self.opt.lambda2
                # print(tmp)
                tmp[torch.isnan(tmp)] = 0.8
                # print(tmp)
                output = F.elu(output) * (tmp)

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
