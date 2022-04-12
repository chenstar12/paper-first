import torch.nn as nn
import torch
import time


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model_name='BaseModel'

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
