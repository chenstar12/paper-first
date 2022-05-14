import json
import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from operator import itemgetter
import gensim
from collections import defaultdict
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import logging
import torch  # 防止gpu断开
from utils import *


def numerize(data):
    uid = list(map(lambda x: user2id[x], data['user_id']))
    iid = list(map(lambda x: item2id[x], data['item_id']))
    data['user_id'] = uid
    data['item_id'] = iid
    return data


if __name__ == '__main__':
    logger = logging.getLogger('')

    t01 = torch.zeros(1, 1).cuda()  # 防断gpu

    start_time = time.time()
    assert (len(sys.argv) >= 2)
    filename = sys.argv[1]

    yelp_data = False
    if len(sys.argv) > 2 and sys.argv[2] == 'yelp':
        print('...............................yelp.......................................')
        # yelp dataset
        yelp_data = True
        save_folder = '../dataset/' + filename[:4] + "_data"  # yelp_data
        data_name = filename[:4]
    else:
        # amazon dataset
        raise RuntimeError('............................No yelp...........................')

    log_file_name = os.path.join('/content/drive/MyDrive/log/dataset',
                                 data_name + time.strftime("-%m%d-%H%M%S", time.localtime()) + '.txt')
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - : %(message)s')
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"数据集名称：{save_folder}")

    if not os.path.exists(save_folder + '/train'):
        os.makedirs(save_folder + '/train')
    if not os.path.exists(save_folder + '/val'):
        os.makedirs(save_folder + '/val')
    if not os.path.exists(save_folder + '/test'):
        os.makedirs(save_folder + '/test')

    if len(PRE_W2V_BIN_PATH) == 0:
        print("Warning: the word embedding file is not provided, will be initialized randomly")
    file = open(filename, errors='ignore')

    logger.info(f"{now()}: Step1: loading raw review datasets...")

    users_id = []
    items_id = []
    ratings = []
    reviews = []
    # sentiment特征：
    polarity = []
    subjectivity = []
    # vader待完成:
    compound = []
    analyzer = SentimentIntensityAnalyzer()
    d = dict()

    if yelp_data:
        for line in file:
            js = json.loads(line)
            if str(js['user_id']) == 'unknown':
                print("unknown user id")
                continue
            if str(js['business_id']) == 'unknown':
                print("unkown item id")
                continue
            try:
                uid = str(js['user_id'])
                iid = str(js['business_id'])
                if (uid, iid) not in d.keys():
                    d[uid + iid] = 1
                else:
                    d[uid + iid] += 1
            except:
                continue

    print('1. searching < 5 ..............................')
    print(len(d))
    for k, v in d.copy().items():
        if v < 5:
            d.pop(k)
        else:
            print(v)

    print(len(d))
    print('2. start processing ..............................')
    if yelp_data:
        for line in file:
            js = json.loads(line)
            if str(js['user_id']) == 'unknown':
                print("unknown user id")
                continue
            if str(js['business_id']) == 'unknown':
                print("unkown item id")
                continue

            uid = str(js['user_id'])
            iid = str(js['business_id'])
            if uid + iid not in d.keys():
                continue
            reviews.append(js['text'])
            users_id.append(uid)
            items_id.append(iid)
            ratings.append(str(js['stars']))

            blob = TextBlob(js['text'])
            pola = blob.sentiment.polarity
            pola = int(pola * 10000)  # 防止被floatTensor截断
            polarity.append(pola)
            subj = blob.sentiment.subjectivity
            subj = int(subj * 10000)
            subjectivity.append(subj)
            compound.append(0)

data_frame = {'user_id': pd.Series(users_id), 'item_id': pd.Series(items_id),
              'ratings': pd.Series(ratings), 'reviews': pd.Series(reviews),
              'polarity': pd.Series(polarity), 'subjectivity': pd.Series(subjectivity)
              }
data = pd.DataFrame(data_frame)  # [['user_id', 'item_id', 'ratings', 'reviews']]
print('3. finally.....................data.shape: ', data.shape)
'''
yelp数据集: 5-core处理
'''
# if yelp_data:
# df_u = data.groupby('user_id').count()
# uid = df_u[df_u['item_id'] < 5].index
# print('user with interacted item < 5 index: ', uid)
# print('(1) begin......data.shape: ', data.shape)
# for u in uid:
#     data.drop(data[data['user_id'] == u].index, inplace=True)
# print('(2) user dropped.....data.shape: ', data.shape)
#
# df_i = data.groupby('item_id').count()
# iid = df_i[df_i['user_id'] < 5].index
# print('items with interacted user < 5 index: ', iid)
# for i in iid:
#     data.drop(data[data['item_id'] == i].index, inplace=True)
# print('(3) item dropped.....data.shape', data.shape)

# blob = TextBlob(data_frame['text'])
# pola = blob.sentiment.polarity
# pola = int(pola * 10000)  # 防止被floatTensor截断
# # polarity.append(pola)
# subj = blob.sentiment.subjectivity
# subj = int(subj * 10000)
# # subjectivity.append(subj)
#
# data_frame['polarity'] = pd.Series(polarity)
# data_frame['subjectivity'] = pd.Series(subjectivity)

del users_id, items_id, ratings, reviews, polarity, subjectivity

uidList, iidList = get_count(data, 'user_id'), get_count(data, 'item_id')
userNum_all = len(uidList)
itemNum_all = len(iidList)
logger.info("===============Start:all  rawData size======================")
logger.info(f"dataNum: {data.shape[0]}")
logger.info(f"userNum: {userNum_all}")
logger.info(f"itemNum: {itemNum_all}")
logger.info(f"data densiy: {data.shape[0] / float(userNum_all * itemNum_all):.4f}")
logger.info("===============End: rawData size========================")

user2id = dict((uid, i) for (i, uid) in enumerate(uidList))
item2id = dict((iid, i) for (i, iid) in enumerate(iidList))
data = numerize(data)

logger.info(f"-" * 60)
logger.info(f"{now()} Step2: split datsets into train/val/test, save into npy data")
data_train, data_test = train_test_split(data, test_size=0.2, random_state=1234)
uids_train, iids_train = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
userNum = len(uids_train)
itemNum = len(iids_train)
logger.info("===============Start: no-preprocess: trainData size======================")
logger.info("dataNum: {}".format(data_train.shape[0]))
logger.info("userNum: {}".format(userNum))
logger.info("itemNum: {}".format(itemNum))
logger.info("===============End: no-preprocess: trainData size========================")

'''
train data添加数据（test data移除数据） 
-> 数据处理完成后的训练集不含test set中ui pair对应的评论文本！！！（因为后面user_reviews_dict是用的训练集）
'''
uidMiss = []
iidMiss = []
if userNum != userNum_all or itemNum != itemNum_all:
    for uid in range(userNum_all):
        if uid not in uids_train:
            uidMiss.append(uid)
    for iid in range(itemNum_all):
        if iid not in iids_train:
            iidMiss.append(iid)

uid_index = []
for uid in uidMiss:
    index = data_test.index[data_test['user_id'] == uid].tolist()
    uid_index.extend(index)
data_train = pd.concat([data_train, data_test.loc[uid_index]])

iid_index = []
for iid in iidMiss:
    index = data_test.index[data_test['item_id'] == iid].tolist()
    iid_index.extend(index)
data_train = pd.concat([data_train, data_test.loc[iid_index]])

all_index = list(set().union(uid_index, iid_index))
data_test = data_test.drop(all_index)

# split validate set and test set
data_test, data_val = train_test_split(data_test, test_size=0.5, random_state=1234)
uidList_train, iidList_train = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
userNum = len(uidList_train)
itemNum = len(iidList_train)
logger.info("===============Start--process finished: trainData size======================")
logger.info("dataNum: {}".format(data_train.shape[0]))
logger.info("userNum: {}".format(userNum))
logger.info("itemNum: {}".format(itemNum))
logger.info("===============End-process finished: trainData size========================")


def extract(data_dict):
    x = []
    y = []
    for i in data_dict.values:
        uid = i[0]
        iid = i[1]
        x.append([uid, iid])
        y.append(float(i[2]))
    return x, y


x_train, y_train = extract(data_train)
x_val, y_val = extract(data_val)
x_test, y_test = extract(data_test)


def extract_sentiment(data_dict):
    senti = []
    for i in data_dict.values:
        senti.append([i[4], i[5], i[6]])
    return senti


s_train = extract_sentiment(data_train)
# s_test = extract_sentiment(data_test)
# s_val = extract_sentiment(data_val)
np.save(f"{save_folder}/train/S_Train.npy", s_train)
# np.save(f"{save_folder}/test/S_Test.npy", s_test)
# np.save(f"{save_folder}/val/S_Val.npy", s_val)

np.save(f"{save_folder}/train/Train.npy", x_train)
np.save(f"{save_folder}/train/Train_Score.npy", y_train)
np.save(f"{save_folder}/val/Val.npy", x_val)
np.save(f"{save_folder}/val/Val_Score.npy", y_val)
np.save(f"{save_folder}/test/Test.npy", x_test)
np.save(f"{save_folder}/test/Test_Score.npy", y_test)

logger.info(now())
logger.info(f"Train data size: {len(x_train)}")
logger.info(f"Val data size: {len(x_val)}")
logger.info(f"Test data size: {len(x_test)}")

logger.info(f"-" * 60)
logger.info(f"{now()} Step3: Construct the vocab and user/item reviews from training set.")
# 2: build vocabulary only with train dataset
user_reviews_dict = {}
item_reviews_dict = {}
user_iid_dict = {}
item_uid_dict = {}
user_len = defaultdict(int)
item_len = defaultdict(int)
# 抽取特征 ---- sentiment:
user_sentiments_dict = {}
item_sentiments_dict = {}

for i in data_train.values:  # 关键！！！用train set获取user_reviews_dict！！！
    str_review = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))

    if len(str_review.strip()) == 0:
        str_review = "<unk>"

    if i[0] in user_reviews_dict:
        user_reviews_dict[i[0]].append(str_review)
        user_iid_dict[i[0]].append(i[1])
    else:
        user_reviews_dict[i[0]] = [str_review]
        user_iid_dict[i[0]] = [i[1]]

    if i[1] in item_reviews_dict:
        item_reviews_dict[i[1]].append(str_review)
        item_uid_dict[i[1]].append(i[0])
    else:
        item_reviews_dict[i[1]] = [str_review]
        item_uid_dict[i[1]] = [i[0]]
    # sentiment:
    if i[0] not in user_sentiments_dict:
        user_sentiments_dict[i[0]] = [[i[4], i[5], i[6]]]
    else:
        user_sentiments_dict[i[0]].append([i[4], i[5], i[6]])

    if i[1] not in item_sentiments_dict:
        item_sentiments_dict[i[1]] = [[i[4], i[5], i[6]]]
    else:
        item_sentiments_dict[i[1]].append([i[4], i[5], i[6]])

# np.save(f"{save_folder}/train/userReview2Sentiment.npy", user_sentiments_dict)
# np.save(f"{save_folder}/train/itemReview2Sentiment.npy", item_sentiments_dict)

vocab, user_review2doc, item_review2doc, user_reviews_dict, item_reviews_dict = build_doc(user_reviews_dict,
                                                                                          item_reviews_dict)
word_index = {}
word_index['<unk>'] = 0
for i, w in enumerate(vocab.keys(), 1):
    word_index[w] = i
logger.info(f"The vocab size: {len(word_index)}")
logger.info(f"Average user document length: {sum([len(i) for i in user_review2doc]) / len(user_review2doc)}")
logger.info(f"Average item document length: {sum([len(i) for i in item_review2doc]) / len(item_review2doc)}")

logger.info(now())
u_minNum, u_maxNum, u_averageNum, u_maxSent, u_minSent, u_pReviewLen, u_pSentLen = countNum(user_reviews_dict)
logger.info("用户最少有{}个评论,最多有{}个评论，平均有{}个评论, " \
            "句子最大长度{},句子的最短长度{}，" \
            "设定用户评论个数为{}： 设定句子最大长度为{}".format(u_minNum, u_maxNum, u_averageNum, u_maxSent, u_minSent, u_pReviewLen,
                                              u_pSentLen))
i_minNum, i_maxNum, i_averageNum, i_maxSent, i_minSent, i_pReviewLen, i_pSentLen = countNum(item_reviews_dict)
logger.info("商品最少有{}个评论,最多有{}个评论，平均有{}个评论," \
            "句子最大长度{},句子的最短长度{}," \
            ",设定商品评论数目{}, 设定句子最大长度为{}".format(i_minNum, i_maxNum, i_averageNum, u_maxSent, i_minSent, i_pReviewLen,
                                              i_pSentLen))
logger.info("最终设定句子最大长度为(取最大值)：{}".format(max(u_pSentLen, i_pSentLen)))
# ########################################################################################################
maxSentLen = max(u_pSentLen, i_pSentLen)
minSentlen = 1

logger.info(f"-" * 60)
logger.info(f"{now()} Step4: padding all the text and id lists and save into npy.")


# 把单个user/item的的review数统一为10/27
def padding_text(textList, num):
    new_textList = []
    if len(textList) >= num:
        new_textList = textList[:num]
    else:
        padding = [[0] * len(textList[0]) for _ in range(num - len(textList))]
        new_textList = textList + padding
    return new_textList


# 把单个user/item的的sentiment数统一为10/27
def padding_sentiment(sentiList, num):
    new_list = []
    if len(sentiList) >= num:
        new_list = sentiList[:num]
    else:
        padding = [[0.0, 0.0, 0.0] for _ in range(num - len(sentiList))]
        new_list = sentiList + padding
    return new_list


def padding_ids(iids, num, pad_id):
    if len(iids) >= num:
        new_iids = iids[:num]
    else:
        new_iids = iids + [pad_id] * (num - len(iids))
    return new_iids


def padding_doc(doc):
    pDocLen = DOC_LEN
    new_doc = []
    for d in doc:
        if len(d) < pDocLen:
            d = d + [0] * (pDocLen - len(d))
        else:
            d = d[:pDocLen]
        new_doc.append(d)

    return new_doc, pDocLen


# 关键逻辑！！！
userReview2Index = []
userDoc2Index = []
user_iid_list = []
userReview2Sentiment = []
analyzer = SentimentIntensityAnalyzer()

for i in range(userNum):

    count_user = 0
    dataList = []
    a_count = 0

    textList = user_reviews_dict[i]
    sentimentList = user_sentiments_dict[i]
    u_iids = user_iid_dict[i]
    u_reviewList = []

    user_iid_list.append(padding_ids(u_iids, u_pReviewLen, itemNum + 1))
    doc2index = [word_index[w] for w in user_review2doc[i]]
    vs = analyzer.polarity_scores(user_review2doc[i])
    doc2index[-1] = int(vs['compound'] * 10000)  # 把sentiment存放在最后一位

    for text in textList:
        text2index = []
        wordTokens = text.strip().split()
        if len(wordTokens) == 0:
            wordTokens = ['<unk>']
        text2index = [word_index[w] for w in wordTokens]  # 每个单词转为vocab对应的索引值
        if len(text2index) < maxSentLen:
            text2index = text2index + [0] * (maxSentLen - len(text2index))
        else:
            text2index = text2index[:maxSentLen]
        u_reviewList.append(text2index)

    userReview2Index.append(padding_text(u_reviewList, u_pReviewLen))
    userDoc2Index.append(doc2index)
    userReview2Sentiment.append(padding_sentiment(user_sentiments_dict[i], u_pReviewLen))  # sentiment

# userReview2Index = []
userDoc2Index, userDocLen = padding_doc(userDoc2Index)
logger.info(f"user document length: {userDocLen}")

itemReview2Index = []
itemDoc2Index = []
item_uid_list = []
itemReview2Sentiment = []
for i in range(itemNum):
    count_item = 0
    dataList = []
    textList = item_reviews_dict[i]
    sentimentList = item_sentiments_dict[i]

    i_uids = item_uid_dict[i]
    i_reviewList = []  # 待添加
    i_reviewLen = []  # 待添加
    item_uid_list.append(padding_ids(i_uids, i_pReviewLen, userNum + 1))
    doc2index = [word_index[w] for w in item_review2doc[i]]
    vs = analyzer.polarity_scores(item_review2doc[i])
    doc2index[-1] = int(vs['compound'] * 10000)  # 把sentiment存放在最后一位

    for text in textList:
        text2index = []
        wordTokens = text.strip().split()
        if len(wordTokens) == 0:
            wordTokens = ['<unk>']
        text2index = [word_index[w] for w in wordTokens]
        if len(text2index) < maxSentLen:
            text2index = text2index + [0] * (maxSentLen - len(text2index))
        else:
            text2index = text2index[:maxSentLen]
        if len(text2index) < maxSentLen:
            text2index = text2index + [0] * (maxSentLen - len(text2index))
        i_reviewList.append(text2index)
    itemReview2Index.append(padding_text(i_reviewList, i_pReviewLen))
    itemDoc2Index.append(doc2index)
    itemReview2Sentiment.append(padding_sentiment(item_sentiments_dict[i], i_pReviewLen))  # sentiment

itemDoc2Index, itemDocLen = padding_doc(itemDoc2Index)
logger.info(f"item document length: {itemDocLen}")

logger.info("-" * 60)
logger.info(f"{now()} start writing npy...")
np.save(f"{save_folder}/train/userReview2Index.npy", userReview2Index)
np.save(f"{save_folder}/train/user_item2id.npy", user_iid_list)
np.save(f"{save_folder}/train/userDoc2Index.npy", userDoc2Index)
np.save(f"{save_folder}/train/userReview2Sentiment.npy", userReview2Sentiment)

np.save(f"{save_folder}/train/itemReview2Index.npy", itemReview2Index)
np.save(f"{save_folder}/train/item_user2id.npy", item_uid_list)
np.save(f"{save_folder}/train/itemDoc2Index.npy", itemDoc2Index)
np.save(f"{save_folder}/train/itemReview2Sentiment.npy", itemReview2Sentiment)

logger.info(f"{now()} write finised")

# #####################################################3,产生w2v############################################
logger.info("-" * 60)
logger.info(f"{now()} Step5: start word embedding mapping...")
vocab_item = sorted(word_index.items(), key=itemgetter(1))
w2v = []
out = 0
if PRE_W2V_BIN_PATH:
    pre_word2v = gensim.models.KeyedVectors.load_word2vec_format(PRE_W2V_BIN_PATH, binary=True)
else:
    pre_word2v = {}
logger.info(f"{now()} 开始提取embedding")
for word, key in vocab_item:
    if word in pre_word2v:
        w2v.append(pre_word2v[word])
    else:
        out += 1
        w2v.append(np.random.uniform(-1.0, 1.0, (300,)))
logger.info("############################")
logger.info(f"out of vocab: {out}")
# print(w2v[1000])
logger.info(f"w2v size: {len(w2v)}")
logger.info("#" * 100)
w2vArray = np.array(w2v)
logger.info(w2vArray.shape)
np.save(f"{save_folder}/train/w2v.npy", w2v)
end_time = time.time()
logger.info(f"{now()} all steps finised, cost time: {end_time - start_time:.4f}s")
