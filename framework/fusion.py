import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    '''
    Fusion Layer for user feature and item feature:
    - sum
    - add
    - concatenation
    - self attention
    '''

    def __init__(self, opt):
        super(FusionLayer, self).__init__()
        if opt.self_att:
            self.attn = SelfAtt(opt.id_emb_size, opt.num_heads)
        self.opt = opt
        self.linear = nn.Linear(opt.feature_dim, opt.feature_dim)  # [] -> [feature_dim,feature_dim]
        self.drop_out = nn.Dropout(0.5)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)  # 初始化权重
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, u_out, i_out):
        # 先：review和id融合
        if self.opt.self_att:
            out = self.attn(u_out, i_out)
            s_u_out, s_i_out = torch.split(out, out.size(1) // 2, 1)
            u_out = u_out + s_u_out
            i_out = i_out + s_i_out
        if self.opt.r_id_merge == 'cat':  # 默认用cat
            u_out = u_out.reshape(u_out.size(0), -1)  # 打平，如：deepConn的[128,1,32]变为torch.Size([128, 32])
            i_out = i_out.reshape(i_out.size(0), -1)  # NARRE的两个[128,2,32]都变成[128,64]
        else:
            u_out = u_out.sum(1)
            i_out = i_out.sum(1)

        # 后：user和item的融合
        if self.opt.ui_merge == 'cat':
            out = torch.cat([u_out, i_out], 1)  # torch.Size([128, 64])
        elif self.opt.ui_merge == 'add':  # NARRE的两个[128,64]进行dot还是[128,64]
            out = u_out + i_out
        else:
            out = u_out * i_out

        return out


class SelfAtt(nn.Module):
    '''
    self attention for interaction
    '''

    def __init__(self, dim, num_heads):
        super(SelfAtt, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(dim, num_heads, 128, 0.4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 1)

    def forward(self, user_fea, item_fea):
        fea = torch.cat([user_fea, item_fea], 1).permute(1, 0, 2)  # batch * 6 * 64
        out = self.encoder(fea)
        return out.permute(1, 0, 2)
