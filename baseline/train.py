import time
import random
import math
import fire

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os

from dataset import ReviewData
import baseline.models as models
import config


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def train(**kwargs):
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Video_Games_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)

    log_file_name = os.path.join('/content/drive/MyDrive/paper-first/log-baseline', opt.dataset[:4] + '-' +
                                 opt.model + '-' + str(time.strftime('%d%H%M')) + '.txt')
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - : %(message)s')
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('\n' + 'model: ' + opt.model + '\n' + 'dataset: ' + opt.dataset + '\n' +
                'batch_size:' + str(opt.batch_size) + '\n' + 'num_epochs: ' + str(opt.num_epochs) + '\n' +
                'r_id_merge: ' + opt.r_id_merge + '\n' + 'ui_merge: ' + opt.ui_merge + '\n' +
                'output: ' + opt.output + '\n' + 'lr: ' + str(opt.lr) + '\n' + 'early_stop: ' + str(opt.early_stop))
    logger.info('alpha: ' + str(opt.alpha))

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.cuda.set_device(opt.gpu_id)

    Net = getattr(models, opt.model)
    model = Net(opt).cuda()

    # 3 data
    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)

    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info(f'train data: {len(train_data)}; test data: {len(val_data)}')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # training
    logger.info("start training.........................................................")
    best_res = 1e+10
    mse_func = nn.MSELoss()

    iter_loss = []  # ??????iteration???loss???????????????
    epoch_train_mse = []
    epoch_val_mse = []
    num_decline = 0  # early_stop ??????
    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        model.train()
        logger.info(f"{now()}  Epoch {epoch}...")
        print(f"{now()}  Epoch {epoch}...")
        for idx, (train_datas, scores) in enumerate(train_data_loader):
            scores = torch.FloatTensor(scores).cuda()

            train_datas = unpack_input(opt, train_datas)  # ?????????????????????????????????reviews, ids, doc

            optimizer.zero_grad()
            output = model(train_datas)

            mse_loss = mse_func(output, scores)
            total_loss += mse_loss.item() * len(scores)  # mse_loss?????????mean
            iter_loss.append(mse_loss.item() * len(scores))

            loss = mse_loss

            loss.backward()
            optimizer.step()

        scheduler.step()

        mse = total_loss * 1.0 / len(train_data)  # total_loss???????????????0??? len(train_data)?????????
        epoch_train_mse.append(mse)
        logger.info(f"\ttrain loss:{total_loss:.4f}, mse: {mse:.4f};")

        # ?????????????????????????????????????????????????????????NDCG???Diversity,MRR,HR,AUC,
        val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
        epoch_val_mse.append(val_mse)

        if val_mse < best_res:
            num_decline = 0  # early_stop ??????
            best_res = val_mse
            logger.info('current best_res: ' + str(best_res) + ', num_decline: ' + str(num_decline))
            model.save(name=opt.dataset, opt=opt.print_opt)
            logger.info("model save")
        else:
            num_decline += 1
            logger.info('current best_res: ' + str(best_res) + ', num_decline: ' + str(num_decline))
            if num_decline >= opt.early_stop:
                logger.info(
                    '=======================Early Stop: ' + 'num_decline = ' + str(num_decline) + '==================')
                break
        logger.info("*" * 30)

    logger.info("-" * 150)
    logger.info(f"{now()} {opt.dataset} {opt.print_opt} best_res:  {best_res}")
    logger.info("-" * 150)
    logger.info('train iteration loss list: ' + str(iter_loss))
    logger.info('epoch_val_mse list: ' + str(epoch_val_mse))
    logger.info('train loss list: ' + str(epoch_train_mse))


def test(**kwargs):
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Video_Games_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    assert (len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.cuda.set_device(opt.gpu_id)

    Net=opt.model
    model = Net(opt).cuda()

    model.load(opt.pth_path)
    logger.info(f"load model: {opt.pth_path}")
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    logger.info(f"{now()}: test in the test dataset")
    predict(model, test_data_loader, opt)


def predict(model, data_loader, opt):
    total_loss = 0.0
    total_maeloss = 0.0
    model.eval()
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            scores = torch.FloatTensor(scores).cuda()
            test_data = unpack_input(opt, test_data)

            output = model(test_data)

            mse_loss = torch.sum((output - scores) ** 2)
            total_loss += mse_loss.item()

            mae_loss = torch.sum(abs(output - scores))
            total_maeloss += mae_loss.item()

    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len

    logger.info(f"evaluation result: mse: {mse:.4f}; rmse: {math.sqrt(mse):.4f}; mae: {mae:.4f};")
    model.train()
    return total_loss, mse, mae


def unpack_input(opt, x):  # ????????????batch????????????
    uids, iids = list(zip(*x))  # ?????????????????? -> ??????????????????????????????tensor???
    uids = list(uids)  # uids????????????????????????tensor???
    iids = list(iids)

    user_reviews = opt.users_review_list[uids]  # ????????????user???reviews
    user_item2id = opt.user2itemid_list[uids]  # ????????????user?????????item id
    user_doc = opt.user_doc[uids]

    item_reviews = opt.items_review_list[iids]
    item_user2id = opt.item2userid_list[iids]
    item_doc = opt.item_doc[iids]

    data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
    data = list(map(lambda x: torch.LongTensor(x).cuda(), data))  # ???data???????????????x??????????????????LongTensor
    return data


if __name__ == "__main__":
    logger = logging.getLogger('')
    fire.Fire()
