# -*- coding: utf-8 -*-
from numpy import *
import os
import pickle
import jieba
import jieba.posseg
import pandas as pd
import numpy as np
import sys
import re
import codecs
from gensim.models import Word2Vec
from processing import cut_sub
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
# reload(sys)
# sys.setdefaultencoding('utf-8')

def data2list(train, test):
    vocab = defaultdict(float)
    all_text = list(train)+list(test)
    for sentence in all_text:
        for word in sentence:
            vocab[word] += 1
    return vocab, all_text

# 将分词好的英文单词相连接
def join_list(list):
    sten = ''
    for item in list[1:]:
        sten +=item+' '
    # 去掉句子最后面的空格
    return sten.rstrip(' ')

def add_unknown_words(word_vecs, vocab, min_df=1, k=32):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def load_data(data):
    print('load_data')
    list_word2ve = []
    for index, item in data.iterrows():
        # 建立词典
        line = ' '.join(jieba.cut(cut_sub(item['content']))) + ' '.join(jieba.cut(cut_sub(item['comment_all'])))
        list_1 = bulid_input(line)
        if list_1:
            list_word2ve.append(list_1)
    return list_word2ve


def bulid_input(line):
    if line == 'Null':
        pass
    else:
        list_1 = []
        for word in str(line).split():
            # 用于去除停用词
            # if word not in stopwords.words('english'):
            list_1.append(word)
        return list_1

# 训练word2vec Model
def word2vec(vocab, all_text):
    # 打开持久化向量
    print('start to word2vec')
    # 训练模型
    word_embedding_path = "embedding\word_embedding.plk"
    w2v = Word2Vec(all_text, size=32, window=4, min_count=1, workers=4)

    temp = {}
    for word in w2v.wv.vocab:
        temp[word] = w2v[word]
    w2v = temp
    pickle.dump(w2v, open(word_embedding_path, 'wb+'))
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    return w2v

def get_W(word_vecs, k=32):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    # vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(len(word_vecs) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map
    

if __name__ == '__main__':
    train_df = pd.read_csv('data/train.csv',
                         skiprows=[0],
                         dtype={'picture_lists':str},
                         names=['id', 'content', 'picture_lists', 'category', 'ncw', 'fake', 'real','comment_2', 'comment_all']
                         )
    test_df = pd.read_csv('data/test_dataset_update.csv',
                          skiprows=[0],
                          dtype={'content': str, 'picture_lists': str},
                          names=['id', 'content', 'picture_lists', 'category', 'comment_2', 'comment_all']
                          )
    train = load_data(train_df)
    test = load_data(test_df)
    vocab, all_text = data2list(train, test)
    max_l = len(max(all_text, key=len))

    w2v = word2vec(vocab, all_text)
    add_unknown_words(w2v, vocab)
    file_path = "data/event_clustering.pickle"

    data = []
    for l in train:
        line_data = []
        for word in l:
            line_data.append(w2v[word])
        line_data = np.matrix(line_data)
        line_data = np.array(np.mean(line_data, 0))[0]
        data.append(line_data)

    data = np.array(data)

    cluster = AgglomerativeClustering(n_clusters=9, affinity='cosine', linkage='complete')
    cluster.fit(data)
    y = np.array(cluster.labels_)
    pickle.dump(y, open(file_path, 'wb+'))

    print("Event length is " + str(len(y)))
    center_count = {}
    for k, i in enumerate(y):
        if i not in center_count:
            center_count[i] = 1
        else:
            center_count[i] += 1
    print(center_count)
    train_df['event_label'] = y

    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2 = rand_vecs = {}
    pickle.dump([W, W2, word_idx_map, vocab, max_l], open("data/word_embedding.pickle", "wb"))
    print("dataset created!")

