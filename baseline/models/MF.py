import torch
import torch.nn as nn
from baseline.utils import BaseModel
import torch.nn.functional as F


class MF(BaseModel):

    def __init__(self, opt):
        super(MF, self).__init__()
        self.opt = opt

        self.user_id_embs = nn.Embedding(opt.user_num, 10000)
        self.item_id_embs = nn.Embedding(opt.item_num, 10000)

        nn.init.normal_(self.user_id_embs.weight)
        nn.init.normal_(self.item_id_embs.weight)

    def forward(self, datas):
        _, _, uids, iids, _, _, _, _ = datas

        user_id_embedding = self.user_id_embs(uids)
        item_id_embedding = self.item_id_embs(iids)

        output = torch.mul(user_id_embedding, item_id_embedding).sum(dim=1)  # torch.mul矩阵点乘 == a * b

        return F.relu(output)
