from baseline.utils import BaseModel
import torch.nn as nn
import torch

from framework.prediction import FM


class FM(BaseModel):
    def __init__(self, opt):
        super(FM, self).__init__()
        self.opt = opt

        self.user_bias = nn.Parameter(torch.FloatTensor([0.1 for _ in range(opt.user_num)]))
        self.item_bias = nn.Parameter(torch.FloatTensor([0.1 for _ in range(opt.item_num)]))
        self.global_bias = nn.Parameter(torch.FloatTensor([4.0]))

        latent_size = opt.id_emb_size

        self.user_embedding = nn.Embedding(opt.user_num, latent_size)
        self.item_embedding = nn.Embedding(opt.user_num, latent_size)

        self.dropout = nn.Dropout(opt.drop_out)

        self.projection = nn.Sequential(
            nn.Dropout(opt.drop_out),
            nn.Linear(2 * latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size)
        )

        self.final = FM(2 * opt.id_emb_size, opt.id_emb_size)

    def forward(self, datas):
        _, _, uids, iids, _, _, _, _ = datas

        # Embed Latent space
        user = self.dropout(self.user_embedding(uids.view(-1)))  # []
        item = self.dropout(self.item_embedding(iids.view(-1)))  # []

        mf_vector = user * item
        cat = torch.cat([user, item], dim=-1)
        mlp_vector = self.projection(cat)

        # Concatenate and get single score
        cat = torch.cat([mlp_vector, mf_vector], dim=-1)
        rating = self.final(cat)[:, 0].view(uids.shape)  # []

        # For the FM
        user_bias = self.user_bias.gather(0, uids.view(-1)).view(uids.shape)
        item_bias = self.item_bias.gather(0, iids.view(-1)).view(iids.shape)
        return user_bias + item_bias + self.global_bias + rating
