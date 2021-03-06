import time
import random
import math
import fire

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import RankReviewData
from framework import Model
import models
import config
from config.utils import *

import logging
import os


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    user, pos, neg = zip(*batch)
    return user, pos, neg


def collate_fn_eval(batch):
    user, pos, neg, scores = zip(*batch)
    return user, pos, neg, scores


def train(**kwargs):
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
                'output: ' + opt.output + '\n' + 'lr: ' + str(opt.lr) + '\n' + 'lambda1: ' +
                str(opt.lambda1) + '\n' + 'inference: ' + str(opt.inference))

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    path_checkpoint = '/content/drive/MyDrive/checkpoints/' + opt.model + '_' + opt.dataset + str(
        int(opt.lambda1 * 100)) + '.pth'
    if os.path.exists(path_checkpoint):
        print('loading exist model......................')
        checkpoint = torch.load(path_checkpoint)
        model = Model(opt, getattr(models, opt.model))  # opt.model: models???????????????DeepDoNN
        model.load_state_dict(checkpoint)
    else:
        print('new a model................................')
        model = Model(opt, getattr(models, opt.model))  # opt.model: models???????????????DeepDoNN

    model.cuda()

    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    # 3 data
    train_data = RankReviewData(opt, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)

    val_data = RankReviewData(opt, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn_eval)

    logger.info(f'train data: {len(train_data)}; test data: {len(val_data)}')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)  # weight_decay:?????????L2?????????
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # training
    logger.info("start training.........................................................")
    best_res = 1e-10

    iter_loss = []  # ??????iteration???loss???????????????
    num_decline = 0  # early_stop ??????
    train_data_len = len(train_data)
    for epoch in range(opt.num_epochs):
        model.train()
        logger.info(f"{now()}  Epoch {epoch}...")
        print(f"{now()}  Epoch {epoch}...")
        for idx, (user, pos_item, neg_item) in enumerate(train_data_loader):
            opt.index = range(idx * (opt.batch_size), min((idx + 1) * (opt.batch_size), train_data_len))

            pos_train_datas = unpack_input_sentiment(opt, list(zip(user, pos_item)))
            neg_train_datas = unpack_input_sentiment(opt, list(zip(user, neg_item)))

            optimizer.zero_grad()

            pos_scores = model(pos_train_datas, opt)
            neg_scores = model(neg_train_datas, opt)

            loss = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
            # opt.stage = 'val'
            # precision, recall, ndcg = predict_ranking(model, val_data_loader, opt)

            loss.backward()
            optimizer.step()

        scheduler.step()

        opt.stage = 'val'
        precision, recall, ndcg = predict_ranking(model, val_data_loader, opt)

        if ndcg > best_res:
            num_decline = 0  # early_stop ??????
            best_res = ndcg
            logger.info('current best_res: ' + str(best_res) + ', num_decline: ' + str(num_decline))
            torch.save(model.state_dict(), path_checkpoint)
            # model.save(name=opt.dataset, opt=opt.print_opt)
            logger.info("model save")
        else:
            num_decline += 1
            logger.info('current best_res: ' + str(best_res) + ', num_decline: ' + str(num_decline))
            if num_decline >= opt.early_stop:
                logger.info('Early Stop: ' + 'num_decline = ' + str(num_decline))
                break
        logger.info("*" * 30)

    logger.info("-" * 150)
    logger.info(f"{now()}  best_res:  {best_res}")
    logger.info("-" * 150)
    logger.info('train iteration loss list: ' + str(iter_loss))


def predict_ranking(model, data_loader, opt):
    print('###########################ranking eval#######################################')
    model.eval()
    with torch.no_grad():
        data_len = len(data_loader.dataset)

        scores_matrix = torch.zeros(opt.user_num, opt.item_num)
        output_matrix = torch.zeros(opt.user_num, opt.item_num)
        for idx, (user, pos_item, neg_item, scores) in enumerate(data_loader):
            opt.index = range(idx * (opt.batch_size), min((idx + 1) * (opt.batch_size), data_len))

            pos_train_datas = unpack_input_sentiment(opt, list(zip(user, pos_item)))
            # neg_train_datas = unpack_input_sentiment(opt, list(zip(user, neg_item)))

            output = model(pos_train_datas, opt)

            for i in range(len(user)):
                output_matrix[user[i], pos_item[i]] = output[i]
                scores_matrix[user[i], pos_item[i]] = scores[i]

        _, index_rank_lists = torch.topk(output_matrix, opt.topk)
        # _, index_scores_matrix = torch.topk(scores_matrix, opt.u_max_r)  # k???????????????100???????????????

        precision = 0.0
        recall = 0.0
        ndcg = 0.0
        diversity = 0.0
        diversity_items = set()

        for i, data in enumerate(opt.user2itemid_list):  # user2itemid_list??????????????????user id
            user = i
            k = opt.topk

            # origin_items_list = index_scores_matrix[user].tolist()
            # ????????????user2itemid_list
            origin_items_list = data.tolist()
            origin_items = set(list(origin_items_list))

            items_list = index_rank_lists[user].tolist()
            items = set(items_list)

            num_hit = len(origin_items.intersection(items))

            precision += float(num_hit / k)
            num_origin_items = len(origin_items_list)
            recall += float(num_hit / num_origin_items)

            diversity_items = diversity_items.union(items)
            diversity = len(diversity_items)

            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(opt.u_max_r, k)):
                max_ndcg_score += 1 / math.log2(i + 2)
            if max_ndcg_score == 0:
                continue

            for i, temp_item in enumerate(items_list[0:k]):
                if temp_item in origin_items:
                    ndcg_score += 1 / math.log2(i + 2)
            ndcg += ndcg_score / max_ndcg_score

            # print('Precision: {:.4f}, Recall: {:.4f}, NDCG: {:.4f}, Diversity: {}'.format(precision, recall, ndcg, diversity))

        data_len = len(data_loader.dataset)
        precision = precision / data_len
        recall = recall / data_len
        ndcg = ndcg / data_len

        logger.info(
            'Precision: {:.4f}, Recall: {:.4f}, NDCG: {:.4f}, Diversity: {}'.format(precision, recall, ndcg, diversity))
        print(
            'Precision: {:.4f}, Recall: {:.4f}, NDCG: {:.4f}, Diversity: {}'.format(precision, recall, ndcg, diversity))
        model.train()
        opt.stage = 'train'
        return precision, recall, ndcg


# ??????????????????????????????
def predict_inference(model, data_loader, opt):
    total_loss_PD = 0.0
    total_loss_PD1 = 0.0
    total_loss_PDA = 0.0
    total_maeloss_PD = 0.0
    total_maeloss_PD1 = 0.0
    total_maeloss_PDA = 0.0
    model.eval()
    with torch.no_grad():
        data_len = len(data_loader.dataset)
        for idx, (test_data, scores) in enumerate(data_loader):
            opt.index = range(idx * (opt.batch_size), min((idx + 1) * (opt.batch_size), data_len))
            scores = torch.FloatTensor(scores).cuda()
            if opt.model[:4] == 'MSCI' or opt.model in ['DeepCoNN1']:  # ??????????????????(??????sentiment??????)
                test_data = unpack_input_sentiment(opt, test_data)
            else:
                test_data = unpack_input(opt, test_data)

            output = model(test_data, opt)

            _, _, _, _, _, _, _, _, _, _, ui_senti = test_data
            po = ui_senti[:, 0] / 10000  # ?????????1???
            sub = ui_senti[:, 1] / 10000  # ?????????2???

            # PD
            output_PD = output + output * opt.lambda1C * po

            # PD1
            output_PD1 = output + output * opt.lambda1C * po * sub

            # PDA
            tmp = po ** opt.lambda2C
            df = pd.DataFrame(tmp.cpu())
            df.fillna(df.mean(), inplace=True)  # ????????????
            tmp = torch.from_numpy(df.values).squeeze(1).cuda()
            output_PDA = output * torch.sigmoid(tmp)  # ??????????????????----sigmoid

            total_loss_PD += torch.sum((output_PD - scores) ** 2).item()
            total_loss_PD1 += torch.sum((output_PD1 - scores) ** 2).item()
            total_loss_PDA += torch.sum((output_PDA - scores) ** 2).item()

            total_maeloss_PD += torch.sum(abs(output_PD - scores)).item()
            total_maeloss_PD1 += torch.sum(abs(output_PD1 - scores)).item()
            total_maeloss_PDA += torch.sum(abs(output_PDA - scores)).item()

    mse = total_loss_PD * 1.0 / data_len
    mse1 = total_loss_PD1 * 1.0 / data_len
    mse2 = total_loss_PDA * 1.0 / data_len
    mae = total_maeloss_PD * 1.0 / data_len
    mae1 = total_maeloss_PD1 * 1.0 / data_len
    mae2 = total_maeloss_PDA * 1.0 / data_len

    logger.info(f"PD  ----- Inference eval: mse: {mse:.4f}; rmse: {math.sqrt(mse):.4f}; mae: {mae:.4f};")
    logger.info(f"PD1 ----- Inference eval: mse: {mse1:.4f}; rmse: {math.sqrt(mse1):.4f}; mae: {mae1:.4f};")
    logger.info(f"PDA ----- Inference eval: mse: {mse2:.4f}; rmse: {math.sqrt(mse2):.4f}; mae: {mae2:.4f};")
    model.train()
    opt.stage = 'train'


if __name__ == "__main__":
    logger = logging.getLogger('')
    opt = None
    fire.Fire()
