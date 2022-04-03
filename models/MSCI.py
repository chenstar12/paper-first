import torch.nn as nn


class MSCI(nn.Module):
    '''
    idea-01：Mitigating Sentiment bias with Causal Intervention
    '''

    def __init__(self, opt):
        super(MSCI, self).__init__()
        self.opt = opt
        self.num_feature = 2  # 0,1,2 = id,doc,review

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
