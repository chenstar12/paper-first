import torch
import torch.nn as nn
from .BaseModel import BaseModel


class MF(BaseModel):

    def __init__(self, opt):
        super(MF, self).__init__()
        self.opt = opt

        self.user_id_embs = nn.Embedding(opt.user_num, 256)
        self.item_id_embs = nn.Embedding(opt.item_num, 256)

        self.reset_para()

    def forward(self, datas):
        _, _, uids, iids, _, _, _, _ = datas

        user_id_embedding = self.user_id_embs(uids)
        item_id_embedding = self.item_id_embs(iids)

        output = torch.mul(user_id_embedding, item_id_embedding).sum(dim=1)

        return output

    def reset_para(self):
        for layer in [self.user_id_embs, self.item_id_embs]:
            nn.init.uniform_(layer.weight, -0.1, 0.1)
