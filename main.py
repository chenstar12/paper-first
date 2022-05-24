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

from dataset import ReviewData
from framework import Model
import models
import config

import logging
import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    # logger.info('\n' + 'lambda1C: ' + str(opt.lambda1C) + '\n' + 'lambda2C: ' +
    #             str(opt.lambda2C))

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    path_checkpoint = '/content/drive/MyDrive/checkpoints/' + opt.model + '_' + opt.dataset + str(
        int(opt.lambda1 * 100)) + '.pth'
    if os.path.exists(path_checkpoint):
        print('loading exist model......................')
        checkpoint = torch.load(path_checkpoint)
        model = Model(opt, getattr(models, opt.model))  # opt.model: models文件夹的如DeepDoNN
        model.load_state_dict(checkpoint)
    else:
        print('new a model................................')
        model = Model(opt, getattr(models, opt.model))  # opt.model: models文件夹的如DeepDoNN

    model = model.cuda()

    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    # 3 data
    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)

    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info(f'train data: {len(train_data)}; test data: {len(val_data)}')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)  # weight_decay:相当于L2正则化
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
    train_data_len = len(train_data)
    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        total_maeloss = 0.0
        model.train()
        logger.info(f"{now()}  Epoch {epoch}...")
        print(f"{now()}  Epoch {epoch}...")
        for idx, (train_datas, scores) in enumerate(train_data_loader):
            scores = torch.FloatTensor(scores).cuda()

            opt.index = range(idx * (opt.batch_size), min((idx + 1) * (opt.batch_size), train_data_len))
            if opt.model[:4] == 'MSCI' or opt.model in ['DeepCoNN1']:  # 获取所有数据(添加sentiment数据)
                train_datas = unpack_input_sentiment(opt, train_datas)
            else:
                train_datas = unpack_input(opt, train_datas)

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

        scheduler.step()
        mse = total_loss * 1.0 / len(train_data)  # total_loss每轮都会置0； len(train_data)：几万
        epoch_train_mse.append(mse)
        logger.info(f"\ttrain loss:{total_loss:.4f}, mse: {mse:.4f};")

        # opt.stage = 'val'
        # predict_ranking(model, val_data_loader, opt)
        # opt.stage = 'val'
        # predict_inference(model, val_data_loader, opt)  # 模仿clickbait：在inference阶段注入sentiment/subjectivity
        opt.stage = 'val'
        val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)

        epoch_val_mse.append(val_mse)

        if val_mse < best_res:
            num_decline = 0  # early_stop 指标
            best_res = val_mse
            logger.info('current best_res: ' + str(best_res) + ', num_decline: ' + str(num_decline))
            torch.save(model.state_dict(), path_checkpoint)
            # model.save(name=opt.dataset, opt=opt.print_opt)
            logger.info("model save")

            # kwargs['pth_path']=
            # test(kwargs)
        else:
            num_decline += 1
            logger.info('current best_res: ' + str(best_res) + ', num_decline: ' + str(num_decline))
            if num_decline >= opt.early_stop:
                logger.info('Early Stop: ' + 'num_decline = ' + str(num_decline))
                break
        logger.info("*" * 30)
    # case study ---- 可视化embedding/representation
    # _, idx = torch.sort(scores)
    # opt.pos_idx = idx[:100]  # pos取最高分100个
    # opt.neg_idx = idx[-100:]  # neg取最低分

    logger.info("-" * 150)
    logger.info(f"{now()}  best_res:  {best_res}")
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
    opt.stage = 'test'

    assert (len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model)).cuda()
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load(opt.pth_path)
    print(f"load model...................: {opt.pth_path}")
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"{now()}: test in the test dataset")
    predict(model, test_data_loader, opt)


# 模型评估
def predict(model, data_loader, opt):
    total_loss = 0.0
    total_maeloss = 0.0
    model.eval()
    with torch.no_grad():
        data_len = len(data_loader.dataset)
        for idx, (test_data, scores) in enumerate(data_loader):
            opt.index = range(idx * (opt.batch_size), min((idx + 1) * (opt.batch_size), data_len))
            scores = torch.FloatTensor(scores).cuda()
            if opt.model[:4] == 'MSCI' or opt.model in ['DeepCoNN1']:  # 获取所有数据(添加sentiment数据)
                test_data = unpack_input_sentiment(opt, test_data)
            else:
                test_data = unpack_input(opt, test_data)

            output = model(test_data, opt)

            mse_loss = torch.sum((output - scores) ** 2)
            total_loss += mse_loss.item()

            mae_loss = torch.sum(abs(output - scores))
            total_maeloss += mae_loss.item()

    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len

    logger.info(f"evaluation result: mse: {mse:.4f}; rmse: {math.sqrt(mse):.4f}; mae: {mae:.4f};")
    print(f"evaluation result: mse: {mse:.4f}; rmse: {math.sqrt(mse):.4f}; mae: {mae:.4f};")

    if opt.stage == 'test':  # 降维，可视化
        def plot_embedding(data, label, title):
            x_min, x_max = np.min(data, 0), np.max(data, 0)
            data = (data - x_min) / (x_max - x_min)

            fig = plt.figure()
            # ax = plt.subplot(111)
            for i in range(data.shape[0]):
                plt.text(data[i, 0], data[i, 1], str(label[i]),
                         color=plt.cm.Set1(label[i] / 10.),
                         fontdict={'weight': 'bold', 'size': 9})
            plt.xticks([])
            plt.yticks([])
            plt.title(title)
            plt.savefig("1.png", dpi=180)
            return fig

        scores = data_loader.dataset.scores
        _, idx = torch.sort(torch.tensor(scores))  # 升序（neg在前）
        neg_idx = idx[:100]
        pos_idx = idx[-100:]
        print(neg_idx)
        # pos = []
        # neg = []
        data = []
        label = []
        for i in pos_idx.numpy().tolist():
            # pos.append(opt.ifea[i])
            data.append(opt.ifea[i])
            label.append(1)
        for i in neg_idx.numpy().tolist():
            # neg.append(opt.ifea[i])
            data.append(opt.ifea[i])
            label.append(0)
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(data)
        fig = plot_embedding(result, label, '')
        plt.show(fig)

    model.train()
    opt.stage = 'train'
    return total_loss, mse, mae


# 不需要了，性能可以了
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
            if opt.model[:4] == 'MSCI' or opt.model in ['DeepCoNN1']:  # 获取所有数据(添加sentiment数据)
                test_data = unpack_input_sentiment(opt, test_data)
            else:
                test_data = unpack_input(opt, test_data)

            output = model(test_data, opt)

            _, _, _, _, _, _, _, _, _, _, ui_senti = test_data
            po = ui_senti[:, 0] / 10000  # 获取第1列
            sub = ui_senti[:, 1] / 10000  # 获取第2列

            # PD
            output_PD = output + output * opt.lambda1C * po

            # PD1
            output_PD1 = output + output * opt.lambda1C * po * sub

            # PDA
            tmp = po ** opt.lambda2C
            df = pd.DataFrame(tmp.cpu())
            df.fillna(df.mean(), inplace=True)  # 均值填充
            tmp = torch.from_numpy(df.values).squeeze(1).cuda()
            output_PDA = output * torch.sigmoid(tmp)  # 新增激活函数----sigmoid

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


def predict_ranking(model, data_loader, opt):
    model.eval()
    with torch.no_grad():
        data_len = len(data_loader.dataset)

        scores_matrix = torch.zeros(opt.user_num, opt.item_num)
        output_matrix = torch.zeros(opt.user_num, opt.item_num)

        for idx, (test_data, scores) in enumerate(data_loader):
            scores = torch.FloatTensor(scores).cuda()
            opt.index = range(idx * (opt.batch_size), min((idx + 1) * (opt.batch_size), data_len))
            if opt.model[:4] == 'MSCI' or opt.model in ['DeepCoNN1']:  # 获取所有数据(添加sentiment数据)
                test_data1 = unpack_input_sentiment(opt, test_data)
            else:
                test_data1 = unpack_input(opt, test_data)

            output = model(test_data1, opt)

            for i in range(len(test_data)):
                output_matrix[test_data[i][0], test_data[i][1]] = output[i]
                scores_matrix[test_data[i][0], test_data[i][1]] = scores[i]

        _, index_rank_lists = torch.topk(output_matrix, opt.topk)
        # _, index_scores_matrix = torch.topk(scores_matrix, opt.u_max_r)  # k待定，先用100，不行再加

        precision = 0.0
        recall = 0.0
        ndcg = 0.0
        diversity = 0.0
        diversity_items = set()

        for i, data in enumerate(opt.user2itemid_list):  # user2itemid_list只是为了获取user id
            user = i

            # origin_items_list = index_scores_matrix[user].tolist()
            # 改动：用user2itemid_list
            origin_items_list = data
            items_list = index_rank_lists[user].tolist()

            k = opt.topk

            items = set(items_list[0:k])
            origin_items_list = origin_items_list[0:k]
            num_origin_items = len(origin_items_list)
            origin_items = set(origin_items_list[0:k])

            num_hit = len(origin_items.intersection(items))
            precision += float(num_hit / k)
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

        data_len = len(data_loader.dataset)
        precision = precision / data_len
        recall = recall / data_len
        ndcg = ndcg / data_len

        logger.info(
            'Precision: {:.4f}, Recall: {:.4f}, NDCG: {:.4f}, Diversity: {}'.format(precision, recall, ndcg, diversity))
        model.train()
        opt.stage = 'train'


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

    if opt.stage == 'train':
        s_train = opt.s_train[opt.index]
        data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc,
                user_sentiments, item_sentiments, s_train]  # 添加了sentiment
    else:
        # s_test = opt.s_test[opt.index]
        data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc,
                user_sentiments, item_sentiments]

    data = list(map(lambda x: torch.LongTensor(x).cuda(), data))  # 将data所有数据表x的类型转换成LongTensor
    return data


if __name__ == "__main__":
    logger = logging.getLogger('')
    opt = None
    fire.Fire()
