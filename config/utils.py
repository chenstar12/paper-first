import torch


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
