import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionLayer(nn.Module):
    '''
        Rating Prediciton(i.e., a regression layer),we pre-define the following rating prediction layers:
        - LFM: Latent Factor Model
        - (N)FM: (Neural) Factorization Machine
        - MLP
        - SUM
    '''

    def __init__(self, opt):
        super(PredictionLayer, self).__init__()
        self.output = opt.output
        if opt.output == "fm":
            self.model = FM(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == "lfm":
            self.model = LFM(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == 'mlp':  # 单层感知机：F.relu(self.fc(feature))
            self.model = MLP(opt.feature_dim)  # feature_dim在models的init中
        elif opt.output == 'nfm':
            self.model = NFM(opt.feature_dim)
        else:
            self.model = torch.sum

    def forward(self, feature, uid, iid):
        if self.output == "lfm" or "fm" or "nfm":
            return self.model(feature, uid, iid)
        else:
            return self.model(feature, 1, keepdim=True)  # NARRE调用MLP的forward函数


class NFM(nn.Module):
    '''
    Neural FM
    '''

    def __init__(self, dim):
        super(NFM, self).__init__()
        self.dim = dim
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(int(dim * 0.5), dim))
        self.mlp = nn.Linear(int(dim * 0.5), int(dim * 0.5))
        self.h = nn.Linear(int(dim * 0.5), 1, bias=False)
        self.drop_out = nn.Dropout(0.5)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0.1)
        nn.init.uniform_(self.fm_V, -0.1, 0.1)
        nn.init.uniform_(self.h.weight, -0.1, 0.1)

    def forward(self, input_vec, *args):
        fm_linear_part = self.fc(input_vec)
        fm_interactions_1 = torch.mm(input_vec, self.fm_V.t())
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2), torch.pow(self.fm_V, 2).t())
        bilinear = 0.5 * (fm_interactions_1 - fm_interactions_2)

        out = F.relu(self.mlp(bilinear))
        out = self.drop_out(out)
        out = self.h(out) + fm_linear_part
        return out


class FM(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(FM, self).__init__()
        self.dim = dim
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(dim, 10))
        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)

    def build_fm(self, input_vec):
        '''
        y = w_0 + \sum {w_ix_i} + \sum_{i=1}\sum_{j=i+1}<v_i, v_j>x_ix_j
        factorization machine layer
        refer: https://github.com/vanzytay/KDD2018_MPCN/blob/master/tylib/lib
                      /compose_op.py#L13
        '''
        # linear part: first two items
        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdim=True) + fm_linear_part
        return fm_output

    def forward(self, feature, uids, iids):
        fm_out = self.build_fm(feature)
        return fm_out + self.b_users[uids] + self.b_items[iids]


class MLP(nn.Module):

    def __init__(self, dim):
        super(MLP, self).__init__()
        self.dim = dim
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, int(dim * 0.5))
        self.fc1 = nn.Linear(int(dim * 0.5), 1)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.bias, a=0, b=0.2)
        nn.init.uniform_(self.fc1.bias, a=0, b=0.2)

    def forward(self, feature, *args, **kwargs):
        return F.relu(self.fc1(F.relu(self.fc(feature))))  # [128,64] -> [128,1], 然后在models中squeeze(1)得到output


class LFM(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(LFM, self).__init__()
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(int(dim * 0.5), 1)
        # -------------------------LFM-user/item-bias-----------------------
        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc.bias, a=0.5, b=1.5)
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)
        nn.init.uniform_(self.b_items, a=0.5, b=1.5)

    def rescale_sigmoid(self, score, a, b):
        return a + torch.sigmoid(score) * (b - a)

    def forward(self, feature, user_id, item_id):
        return self.rescale_sigmoid(self.fc(feature), 1.0, 5.0) + self.b_users[user_id] + self.b_items[item_id]
        # return self.fc(feature) + self.b_users[user_id] + self.b_items[item_id]
