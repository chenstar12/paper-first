import numpy as np


class DefaultConfig:
    model = 'DeepCoNN'
    dataset = ''

    # -------------base config-----------------------#
    use_gpu = True
    gpu_id = 0
    multi_gpu = False
    gpu_ids = []

    seed = 2022
    num_epochs = 200
    num_workers = 0

    optimizer = 'Adam'
    weight_decay = 1e-3
    lr = 2e-3
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
    early_stop = 4  # 在验证集连续4轮mse下降时early stop
    pth_path = ""  # the saved pth path for test
    print_opt = 'default'

    batch_size = 128
    print_step = 100

    vocab_size = 50002
    word_dim = 300

    def set_path(self, name):
        '''
        specific
        '''
        self.data_root = f'./dataset/{name}'
        prefix = f'{self.data_root}/train'

        self.user_list_path = f'{prefix}/userReview2Index.npy'
        self.item_list_path = f'{prefix}/itemReview2Index.npy'

        self.user2itemid_path = f'{prefix}/user_item2id.npy'
        self.item2userid_path = f'{prefix}/item_user2id.npy'

        self.user_doc_path = f'{prefix}/userDoc2Index.npy'
        self.item_doc_path = f'{prefix}/itemDoc2Index.npy'

        self.user_sentiment_path = f'{prefix}/userReview2Sentiment.npy'
        self.item_sentiment_path = f'{prefix}/itemReview2Sentiment.npy'

        self.w2v_path = f'{prefix}/w2v.npy'

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
        self.item_doc = np.load(self.item_doc_path, encoding='bytes')
        self.userReview2Sentiment = np.load(self.user_sentiment_path, encoding='bytes')
        self.itemReview2Sentiment = np.load(self.item_sentiment_path, encoding='bytes')

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
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
