import time
import random
import math
import fire

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ReviewData
from framework import Model
import models
import config

import logging
import os


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def train(**kwargs):
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Video_Games_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)

    log_file_name = os.path.join('/content/drive/MyDrive/log', opt.dataset[:4] + '-' +
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
                'output: ' + opt.output + '\n' + 'lr: ' + str(opt.lr) + '\n' + 'early_stop: ' + str(opt.early_stop) +
                '\n' + 'gamma: ' + str(opt.gamma) + '\n' + 'lambda1: ' + str(opt.lambda1) + '\n' + 'lambda2: ' +
                str(opt.lambda2) + '\n' + 'inference: ' + str(opt.inference))

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))  # opt.model: models文件夹的如DeepDoNN
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    # 3 data
    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)

    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info(f'train data: {len(train_data)}; test data: {len(val_data)}')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)  # 相当于L2正则化
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
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)

            if opt.model[:4] == 'MSCI':  # 获取所有数据(添加sentiment数据)
                train_datas = unpack_input_sentiment(opt, train_datas)
            else:
                train_datas = unpack_input(opt, train_datas)  # 获取所有数据！！！即：reviews, ids, doc

            optimizer.zero_grad()
            output = model(train_datas, opt)

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
            # predict_ranking(model, val_data_loader, opt)

            # if opt.fine_step:  # 默认False。。。。。
            #     if idx % opt.print_step == 0 and idx > 0:
            #         logger.info("\t{}, {} step finised;".format(now(), idx))
            #         val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
            #         if val_loss < min_loss:
            #             model.save(name=opt.dataset, opt=opt.print_opt)
            #             min_loss = val_loss
            #             logger.info("\tmodel save")
            #         if val_loss > min_loss:
            #             best_res = min_loss

        scheduler.step()

        mse = total_loss * 1.0 / len(train_data)  # total_loss每轮都会置0； len(train_data)：几万
        epoch_train_mse.append(mse)
        logger.info(f"\ttrain loss:{total_loss:.4f}, mse: {mse:.4f};")

        # 排序任务的评价指标（不是点击率任务）：NDCG，Diversity,MRR,HR,AUC,
        predict_ranking(model, val_data_loader, opt)
        # opt.stage = 'val'
        val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
        # opt.stage = 'train'
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


# 模型评估
def predict(model, data_loader, opt):
    total_loss = 0.0
    total_maeloss = 0.0
    model.eval()
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)

            if opt.model[:4] == 'MSCI':  # 获取所有数据(添加sentiment数据)
                test_data = unpack_input_sentiment(opt, test_data)
            else:
                test_data = unpack_input(opt, test_data)

            output = model(test_data, opt)

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


# 添加排序指标：ndcg，Diversity,MRR,HR,AUC,recall，acc....
'''
Diversity@K：the number of unique items in all topK recommendation lists 
（可用于干预的方法，但是调参后的方法不适用）
'''


def predict_ranking(model, data_loader, opt):
    model.eval()
    with torch.no_grad():

        scores_matrix = torch.zeros(opt.user_num, opt.item_num)
        output_matrix = torch.zeros(opt.user_num, opt.item_num)

        for idx, (test_data, scores) in enumerate(data_loader):
            scores = torch.FloatTensor(scores).cuda()
            if opt.model[:4] == 'MSCI':  # 获取所有数据(添加sentiment数据)
                test_data1 = unpack_input_sentiment(opt, test_data)
            else:
                test_data1 = unpack_input(opt, test_data)
            output = model(test_data1, opt)

            for i in range(len(test_data)):
                output_matrix[test_data[i][0], test_data[i][1]] = output[i]
                scores_matrix[test_data[i][0], test_data[i][1]] = scores[i]

        _, index_rank_lists = torch.topk(output_matrix, opt.topk[-1])
        _, index_scores_matrix = torch.topk(scores_matrix, opt.item_num)
        print('-' * 100)
        print(index_rank_lists.shape)
        print(output_matrix)
        print(scores_matrix)
        print(index_rank_lists)
        print(index_scores_matrix)

        precision = np.array([0.0] * len(opt.topk))
        recall = np.array([0.0] * len(opt.topk))
        ndcg = np.array([0.0] * len(opt.topk))
        diversity = np.array([0.0] * len(opt.topk))

        for idx, (test_data, scores) in enumerate(data_loader):
            for data in test_data:
                user = data[0]
                origin_items = set(index_scores_matrix[user])
                num_origin_items = len(origin_items)
                items_list = index_rank_lists[user]
                diversity_set = set()
                for ind, k in enumerate(opt.topk):
                    items = set(items_list[0:k])
                    num_hit = len(origin_items.intersection(items))
                    diversity_set = diversity_set.union(set(items_list))
                    print('num_hit: ', num_hit)
                    print('diversity_set: ', len(diversity_set))

                    precision[ind] += float(num_hit / k)
                    recall[ind] += float(num_hit / num_origin_items)
                    diversity[ind] = len(diversity_set)

                    ndcg_score = 0.0
                    max_ndcg_score = 0.0

                    for i in range(min(num_origin_items, k)):
                        max_ndcg_score += 1 / math.log2(i + 2)
                    if max_ndcg_score == 0:
                        continue

                    for i, temp_item in enumerate(items_list[0:k]):
                        if temp_item in origin_items:
                            ndcg_score += 1 / math.log2(i + 2)

                    ndcg[ind] += ndcg_score / max_ndcg_score

        data_len = len(data_loader.dataset)

        precision = precision / data_len
        recall = recall / data_len
        ndcg = ndcg / data_len

        logger.info(
            'Precision: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(precision[0], precision[1], precision[2], precision[3]))
        logger.info('Recall: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(recall[0], recall[1], recall[2], recall[3]))
        logger.info(
            'NDCG: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(ndcg[0], ndcg[1], ndcg[2], ndcg[3]))
        logger.info(
            'Diversity: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(diversity[0], diversity[1], diversity[2], diversity[3]))


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


def unpack_input_sentiment(opt, x):
    uids, iids = list(zip(*x))  # 解包两列数据 -> 两列元组（数据类型：tensor）
    uids = list(uids)  # uids列表（数据类型：tensor）
    iids = list(iids)

    user_reviews = opt.users_review_list[uids]  # 检索出该user的reviews
    user_item2id = opt.user2itemid_list[uids]  # 检索出该user对应的item id
    user_doc = opt.user_doc[uids]
    user_sentiments = opt.userReview2Sentiment[uids]  # sentiment

    item_reviews = opt.items_review_list[iids]
    item_user2id = opt.item2userid_list[iids]
    item_doc = opt.item_doc[iids]
    item_sentiments = opt.itemReview2Sentiment[iids]  # sentiment

    data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc,
            user_sentiments, item_sentiments]  # 添加了sentiment
    data = list(map(lambda x: torch.LongTensor(x).cuda(), data))  # 将data所有数据表x的类型转换成LongTensor
    return data


if __name__ == "__main__":
    logger = logging.getLogger('')
    opt = None
    fire.Fire()

# def test(**kwargs):
#     opt.stage = 'val'
#     if 'dataset' not in kwargs:
#         opt = getattr(config, 'Video_Games_data_Config')()
#     else:
#         opt = getattr(config, kwargs['dataset'] + '_Config')()
#     opt.parse(kwargs)
#     assert (len(opt.pth_path) > 0)
#     random.seed(opt.seed)
#     np.random.seed(opt.seed)
#     torch.manual_seed(opt.seed)
#     if opt.use_gpu:
#         torch.cuda.manual_seed_all(opt.seed)
#
#     if len(opt.gpu_ids) == 0 and opt.use_gpu:
#         torch.cuda.set_device(opt.gpu_id)
#
#     model = Model(opt, getattr(models, opt.model))
#     if opt.use_gpu:
#         model.cuda()
#         if len(opt.gpu_ids) > 0:
#             model = nn.DataParallel(model, device_ids=opt.gpu_ids)
#     if model.net.num_fea != opt.num_fea:
#         raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")
#
#     model.load(opt.pth_path)
#     logger.info(f"load model: {opt.pth_path}")
#     test_data = ReviewData(opt.data_root, mode="Test")
#     test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
#     logger.info(f"{now()}: test in the test dataset")
#     predict_loss, test_mse, test_mae = predict(model, test_data_loader, opt)
