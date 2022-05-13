import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

P_REVIEW = 0.85
MAX_DF = 0.7
MAX_VOCAB = 50000
DOC_LEN = 500
PRE_W2V_BIN_PATH = ""  # the pre-trained word2vec files


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def get_count(data, id):
    ids = set(data[id].tolist())
    return ids


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)  # \s 匹配任何空白字符，包括:空格、制表符、换页符
    string = re.sub(r"\s{2,}", " ", string)  # {m,n} 最少匹配 m 次且最多匹配 n 次。
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()


def bulid_vocbulary(xDict):
    rawReviews = []
    for (id, text) in xDict.items():
        rawReviews.append(' '.join(text))
    return rawReviews


def build_doc(u_reviews_dict, i_reviews_dict):
    '''
    1. extract the vocab
    2. filter the reviews and documents of users and items
    '''
    u_reviews = []
    for ind in range(len(u_reviews_dict)):
        u_reviews.append(' <SEP> '.join(u_reviews_dict[ind]))

    i_reviews = []
    for ind in range(len(i_reviews_dict)):
        i_reviews.append('<SEP>'.join(i_reviews_dict[ind]))

    vectorizer = TfidfVectorizer(max_df=MAX_DF, max_features=MAX_VOCAB)  # max document frequency: 最大词频（去掉无意义的词）
    vectorizer.fit(u_reviews)
    vocab = vectorizer.vocabulary_
    vocab[MAX_VOCAB] = '<SEP>'

    def clean_review(rDict):
        new_dict = {}
        for k, text in rDict.items():
            new_reviews = []
            for r in text:
                words = ' '.join([w for w in r.split() if w in vocab])  # 只保留vocab(词表)中包含的词
                new_reviews.append(words)
            new_dict[k] = new_reviews
        return new_dict

    def clean_doc(raw):
        new_raw = []
        for line in raw:
            review = [word for word in line.split() if word in vocab]
            if len(review) > DOC_LEN:
                review = review[:DOC_LEN]
            new_raw.append(review)
        return new_raw

    u_reviews_dict = clean_review(u_reviews_dict)
    i_reviews_dict = clean_review(i_reviews_dict)

    u_doc = clean_doc(u_reviews)
    i_doc = clean_doc(i_reviews)

    return vocab, u_doc, i_doc, u_reviews_dict, i_reviews_dict


def countNum(xDict):
    minNum = 100
    maxNum = 0
    sumNum = 0
    maxSent = 0
    minSent = 3000
    # pSentLen = 0
    ReviewLenList = []
    SentLenList = []
    for (i, text) in xDict.items():
        sumNum = sumNum + len(text)
        if len(text) < minNum:
            minNum = len(text)
        if len(text) > maxNum:
            maxNum = len(text)
        ReviewLenList.append(len(text))
        for sent in text:
            # SentLenList.append(len(sent))
            if sent != "":
                wordTokens = sent.split()
            if len(wordTokens) > maxSent:
                maxSent = len(wordTokens)
            if len(wordTokens) < minSent:
                minSent = len(wordTokens)
            SentLenList.append(len(wordTokens))
    averageNum = sumNum // (len(xDict))

    x = np.sort(SentLenList)
    xLen = len(x)
    pSentLen = x[int(P_REVIEW * xLen) - 1]
    x = np.sort(ReviewLenList)
    xLen = len(x)
    pReviewLen = x[int(P_REVIEW * xLen) - 1]

    return minNum, maxNum, averageNum, maxSent, minSent, pReviewLen, pSentLen
