from baseline.utils import BaseModel
import torch.nn as nn
import torch


class MF1(BaseModel):
    def __init__(self, opt):
        super(MF1, self).__init__()
        self.opt = opt
        latent_size = opt.id_emb_size

        self.user_embedding = nn.Embedding(opt.user_num, latent_size)
        self.item_embedding = nn.Embedding(opt.user_num, latent_size)

        self.user_bias = nn.Parameter(torch.FloatTensor([0.1 for _ in range(opt.user_num)]))
        self.item_bias = nn.Parameter(torch.FloatTensor([0.1 for _ in range(opt.item_num)]))
        self.global_bias = nn.Parameter(torch.FloatTensor([4.0]))

        self.dropout = nn.Dropout(opt.drop_out)

    def forward(self, datas):
        _, _, uids, iids, _, _, _, _ = datas

        # Embed Latent space
        user = self.dropout(self.user_embedding(uids.view(-1)))  # []
        item = self.dropout(self.item_embedding(iids.view(-1)))  # []

        rating = torch.sum(user * item, dim=-1).view(uids.shape)

        user_bias = self.user_bias.gather(0, uids.view(-1)).view(uids.shape)
        item_bias = self.item_bias.gather(0, iids.view(-1)).view(iids.shape)
        return user_bias + item_bias + self.global_bias + rating
