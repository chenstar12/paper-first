'''
User/Item Representation Learning Layer (models/*.py): the main part of most baseline methods;
such as : the CNN encoder in DeepCoNN;
'''
from .deepconn import DeepCoNN
from .daml import DAML
from .narre import NARRE
from .d_attn import D_ATTN
from .mpcn import MPCN
# mine
from .MSCI import MSCI
from MSCI0 import MSCI0
from .MSCI2 import MSCI2  # 最合理（理论上）
from .MSCI3 import MSCI3
from .MSCI4 import MSCI4
