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
import baseline
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

    log_file_name = os.path.join('/content/drive/MyDrive/log-baseline', opt.dataset[:4] + '-' +
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
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = baseline.models.MF(opt)  # opt.model: models文件夹的如DeepDoNN
    model.cuda()

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
    min_loss = 1e+10
    best_res = 1e+10
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()

    iter_loss = []  # 每个iteration的loss，用来画图
    epoch_train_mse = []
    epoch_val_mse = []
    num_decline = 0  # early_stop 指标
    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        total_maeloss = 0.0
        model.train()
        logger.info(f"{now()}  Epoch {epoch}...")
        print(f"{now()}  Epoch {epoch}...")
        for idx, (train_datas, scores) in enumerate(train_data_loader):
            scores = torch.FloatTensor(scores).cuda()

            train_datas = unpack_input(opt, train_datas)  # 获取所有数据！！！即：reviews, ids, doc

            optimizer.zero_grad()
            output = model(train_datas)

            mse_loss = mse_func(output, scores)
            total_loss += mse_loss.item() * len(scores)  # mse_loss默认取mean
            iter_loss.append(mse_loss.item() * len(scores))

            mae_loss = mae_func(output, scores)
            total_maeloss += mae_loss.item()

            smooth_mae_loss = smooth_mae_func(output, scores)

            if opt.loss_method == 'mse':
                loss = mse_loss
            if opt.loss_method == 'rmse':
                loss = torch.sqrt(mse_loss) / 2.0
            if opt.loss_method == 'mae':
                loss = mae_loss
            if opt.loss_method == 'smooth_mae':
                loss = smooth_mae_loss

            loss.backward()
            optimizer.step()

        scheduler.step()

        mse = total_loss * 1.0 / len(train_data)  # total_loss每轮都会置0； len(train_data)：几万
        epoch_train_mse.append(mse)
        logger.info(f"\ttrain loss:{total_loss:.4f}, mse: {mse:.4f};")

        # 排序任务的评价指标（不是点击率任务）：NDCG，Diversity,MRR,HR,AUC,
        val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
        epoch_val_mse.append(val_mse)

        if val_mse < best_res:
            num_decline = 0  # early_stop 指标
            best_res = val_mse
            logger.info('current best_res: ' + str(best_res) + ', num_decline: ' + str(num_decline))

            model.save(name=opt.dataset, opt=opt.print_opt)
            min_loss = val_loss
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
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = baseline.models.MF(opt).cuda()

    model.load(opt.pth_path)
    logger.info(f"load model: {opt.pth_path}")
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    logger.info(f"{now()}: test in the test dataset")
    predict_loss, test_mse, test_mae = predict(model, test_data_loader, opt)


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

        # 排序任务的评价指标（不是点击率任务）：NDCG，Diversity,MRR,HR,AUC,

    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len

    logger.info(f"evaluation result: mse: {mse:.4f}; rmse: {math.sqrt(mse):.4f}; mae: {mae:.4f};")
    model.train()
    return total_loss, mse, mae


def unpack_input(opt, x):  # 打包一个batch所有数据
    uids, iids = list(zip(*x))  # 解包两列数据 -> 两列元组（数据类型：tensor）
    uids = list(uids)  # uids列表（数据类型：tensor）
    iids = list(iids)

    user_reviews = opt.users_review_list[uids]  # 检索出该user的reviews
    user_item2id = opt.user2itemid_list[uids]  # 检索出该user对应的item id
    user_doc = opt.user_doc[uids]

    item_reviews = opt.items_review_list[iids]
    item_user2id = opt.item2userid_list[iids]
    item_doc = opt.item_doc[iids]

    data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
    data = list(map(lambda x: torch.LongTensor(x).cuda(), data))  # 将data所有数据表x的类型转换成LongTensor
    return data


if __name__ == "__main__":
    logger = logging.getLogger('')

    fire.Fire()