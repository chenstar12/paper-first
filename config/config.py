import numpy as np


class DefaultConfig:
    # pos_u = []
    # pos_i = []
    # neg_u = []
    # neg_i = []
    #
    # pos_idx = []
    # neg_idx = []
    ufea = []
    ifea = []

    model = 'DeepCoNN'
    dataset = ''

    # -------------base config-----------------------#
    use_gpu = True
    gpu_id = 0
    multi_gpu = False
    gpu_ids = []

    seed = 1234
    num_epochs = 200
    num_workers = 0

    optimizer = 'Adam'
    weight_decay = 1e-3
    lr = 1e-3
    loss_method = 'mse'
    drop_out = 0.5

    use_word_embedding = True

    id_emb_size = 32
    query_mlp_size = 128
    fc_dim = 32

    doc_len = 500
    filters_num = 100
    kernel_size = 3

    num_fea = 1  # id feature, review feature, doc feature
    use_review = True
    use_doc = True
    self_att = False

    r_id_merge = 'cat'  # review and ID feature
    ui_merge = 'cat'  # cat/add/dot
    output = 'lfm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    fine_step = False  # save mode in step level, default in epoch
    early_stop = 10  # 在验证集连续10轮mse下降时early stop
    pth_path = ""  # the saved pth path for test
    print_opt = 'default'
    task = 'rank'

    batch_size = 512
    print_step = 100

    vocab_size = 50002
    word_dim = 300

    gamma = 0  # 干预强度，拟取值范围[0.2,0.4,0.6,0.8]

    inference = 'trans-PD1'
    ei = ''
    eval = ''

    # adjusting参数
    lambda1 = 0.5  # PDA减法调参
    lambda2 = 0.01  # PDA指数调参
    lambda1C = 0.01  # 另一个评估函数PD,PD1
    lambda2C = 0.01  # 另一个评估函数PDA

    stage = 'train'  # 模仿clickbait：在评估阶段（test和validation）进行微调
    # topk = [5, 10, 50, 100]  # 排序指标 ---- 与原论文一致
    topk = 50  # 排序指标 ---- 与原论文一致
    index = range(128)  # 用于获取ui-sentiment

    def set_path(self, name):
        '''
        specific
        '''
        self.data_root = f'./dataset/{name}'
        prefix = self.data_root

        self.user_list_path = f'{prefix}/train/userReview2Index.npy'
        self.item_list_path = f'{prefix}/train/itemReview2Index.npy'

        self.user2itemid_path = f'{prefix}/train/user_item2id.npy'
        self.item2userid_path = f'{prefix}/train/item_user2id.npy'

        self.user_doc_path = f'{prefix}/train/userDoc2Index.npy'
        # self.s_u_path = f'{prefix}/train/userDoc2S.npy'
        self.item_doc_path = f'{prefix}/train/itemDoc2Index.npy'
        # self.s_i_path = f'{prefix}/train/itemDoc2S.npy'

        self.user_sentiment_path = f'{prefix}/train/userReview2Sentiment.npy'
        self.item_sentiment_path = f'{prefix}/train/itemReview2Sentiment.npy'

        self.s_train_path = f'{prefix}/train/S_Train.npy'
        self.s_test_path = f'{prefix}/test/S_Test.npy'
        self.s_val_path = f'{prefix}/val/S_Val.npy'

        self.w2v_path = f'{prefix}/train/w2v.npy'

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        print("load npy from dist...")
        self.users_review_list = np.load(self.user_list_path, encoding='bytes')
        self.items_review_list = np.load(self.item_list_path, encoding='bytes')
        self.user2itemid_list = np.load(self.user2itemid_path, encoding='bytes')
        self.item2userid_list = np.load(self.item2userid_path, encoding='bytes')
        self.user_doc = np.load(self.user_doc_path, encoding='bytes')
        # self.s_u = np.load(self.s_u_path, encoding='bytes')
        self.item_doc = np.load(self.item_doc_path, encoding='bytes')
        # self.s_i = np.load(self.s_i_path, encoding='bytes')
        self.userReview2Sentiment = np.load(self.user_sentiment_path, encoding='bytes', allow_pickle=True)
        self.itemReview2Sentiment = np.load(self.item_sentiment_path, encoding='bytes', allow_pickle=True)

        self.s_train = np.load(self.s_train_path, encoding='bytes')
        # self.s_test = np.load(self.s_test_path, encoding='bytes')
        # self.s_val = np.load(self.s_val_path, encoding='bytes')

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('**********************命令行参数/options/hyper_params***************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))
        print('*************************************************')


class Video_Games_data_Config(DefaultConfig):

    def __init__(self):
        self.dataset = 'Video_Games_data'
        self.set_path('Video_Games_data')

    r_max_len = 214

    u_max_r = 10
    i_max_r = 27

    train_data_size = 185439
    test_data_size = 23171
    val_data_size = 23170

    user_num = 24303 + 2
    item_num = 10672 + 2


class Kindle_Store_data_Config(DefaultConfig):

    def __init__(self):
        self.dataset = 'Kindle_Store_data'
        self.set_path('Kindle_Store_data')

    r_max_len = 123

    u_max_r = 16
    i_max_r = 19

    train_data_size = 786159
    test_data_size = 98230
    val_data_size = 98230

    user_num = 68223 + 2
    item_num = 61934 + 2


class Gourmet_Food_data_Config(DefaultConfig):

    def __init__(self):
        self.dataset = 'Gourmet_Food_data'
        self.set_path('Gourmet_Food_data')

    r_max_len = 98

    u_max_r = 12
    i_max_r = 17

    train_data_size = 121003
    test_data_size = 15126
    val_data_size = 15125

    user_num = 14681 + 2
    item_num = 8713 + 2


class Electronics_data_Config(DefaultConfig):

    def __init__(self):
        self.dataset = 'Electronics_data'
        self.set_path('Electronics_data')

    r_max_len = 123

    u_max_r = 10
    i_max_r = 29

    train_data_size = 1351527
    test_data_size = 168830
    val_data_size = 168831

    user_num = 192403 + 2
    item_num = 63001 + 2


class Yelp_data_Config(DefaultConfig):
    def __init__(self):
        self.dataset = 'yelp_data'
        self.set_path('yelp_data')

    # 数据处理未完成；先照抄Electronics
    r_max_len = 123

    u_max_r = 10
    i_max_r = 29

    train_data_size = 1351527
    test_data_size = 168830
    val_data_size = 168831

    user_num = 192403 + 2
    item_num = 63001 + 2


class Pet_Supplies_data_Config(DefaultConfig):

    def __init__(self):
        self.dataset = 'Pet_Supplies_data'
        self.set_path('Pet_Supplies_data')

    r_max_len = 95

    u_max_r = 9
    i_max_r = 22

    train_data_size = 126283
    test_data_size = 15777
    val_data_size = 15776

    user_num = 19856 + 2
    item_num = 8510 + 2
