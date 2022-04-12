import torch
import torch.nn as nn


class MF(nn.Module):

    def __init__(self, opt):
        super(MF, self).__init__()
        self.opt = opt

        self.user_id_embs = nn.Embedding(opt.user_num, opt.id_emb_size)
        self.item_id_embs = nn.Embedding(opt.item_num, opt.id_emb_size)

        self.reset_para()

    def forward(self, datas):
        _, _, uids, iids, _, _, _, _ = datas

        user_id_embedding = self.user_id_embs(uids)
        item_id_embedding = self.item_id_embs(iids)

        output = torch.matmul(user_id_embedding, item_id_embedding)  # predict

        return output

    def reset_para(self):
        for fc in [self.user_id_embs, self.item_id_embs]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)
