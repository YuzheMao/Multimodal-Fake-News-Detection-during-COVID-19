import jieba
import torch
import argparse
import time
import ipdb
import pickle
import pandas as pd
import torch.nn as nn
import numpy as np

from sklearn import metrics
from model import TextCNN, to_var, to_np
from processing import *
from processing import EVENT
from dataset import DatasetWithoutImg
from torch.utils.data import DataLoader

# 超参
parser = argparse.ArgumentParser()

parser.add_argument('--vocab_size', type=int, default=300, help='vocabulary size')
parser.add_argument('--crop_size', type=int, default=112, help='crop size')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dimension')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--embedding_dim', type=int, default=32, help='embedding dimension')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--lambd', type=int, default=1, help='lambda')
parser.add_argument('--dropout', type=int, default=0.5, help='dropout')
parser.add_argument('--event_num', type=int, default=2, help='event number')
args = parser.parse_args()

def select_test(train, selec_indices):
    id_list = []
    text_list = []
    event_list = []
    for i in selec_indices:
        id_list.append(train['id'][i])
        text_list.append(train['text'][i])
        event_list.append(train['event'][i])
    return {'id': id_list, 'text': text_list, 'event': event_list}

def split_for_test(train, percent):
    whole_len = len(train['id'])

    train_indices = (sample(range(whole_len), int(whole_len * percent)))
    train_data = select_test(train, train_indices)
    print("train data size is "+ str(len(train_data['id'])))
    # print()

    validation = select_test(train, np.delete(range(whole_len), train_indices))
    print("validation size is "+ str(len(validation['id'])))
    print("train and validation data set has been splited")

    return train_data, validation

def divide_data(df, mode):
    label_list = []
    id_list = []
    text_list = []
    event_list = []

    for index, item in df.iterrows():
        if not type(item['picture_lists']) == float:
            imgs = cut_imgs(item['picture_lists'])
            # 全取
            for img in imgs:
                if mode == 'train_pictures':
                    label_list.append(get_label(item[TARGET]))
                id_list.append(item['id'])
                text_list.append(item['text_cut'])
                event_list.append(EVENT[item['category']])
        else:
            if mode == 'train_pictures':
                label_list.append(get_label(item[TARGET]))
            id_list.append(item['id'])
            text_list.append(item['text_cut'])
            event_list.append(EVENT[item['category']])
        # 随机取一个
        # img = imgs[random.randint(0, len(imgs) - 1)] if len(imgs) > 1 else imgs[0]
        # img_list.append(img)
        # id_list.append(item['id'])
        # text_list.append(item['text_cut'])
        # event_list.append(EVENT[item['category']])
        # if mode == 'train_pictures':
        #     label_list.append(get_label(item[TARGET]))

        return {'id': id_list, 'text': text_list, 'event': event_list}

# 清洗数据
train_df = pd.read_csv('data/train.csv',
                       skiprows=[0],
                       dtype={'picture_lists':str},
                       names=['id', 'content', 'picture_lists', 'category', 'ncw', 'fake', 'real','comment_2', 'comment_all']
                       )
test_df = pd.read_csv('data/test_dataset_update.csv',
                      skiprows=[0],
                      dtype={'content':str, 'picture_lists':str},
                      names=['id', 'content', 'picture_lists', 'category', 'comment_2', 'comment_all']
                      )
att_df = pd.read_csv('data/label.csv',
                      skiprows=[0],
                       usecols = [0,1,4,7],
                      names=['id', 'fake', 'real', 'ncw']
                      )
att_df = att_df[(att_df['fake'] == 0.5) | (att_df['real'] == 0.5) | (att_df['ncw'] == 0.5) ]
label_df = test_df[test_df.id.isin(att_df['id'])]
test_df = test_df[~test_df.id.isin(att_df['id'])]

test_df.reset_index(drop=True, inplace=True)
label_df.reset_index(drop=True, inplace=True)

label_df['ncw'] = fine_label(att_df['ncw'])
label_df['fake'] = fine_label(att_df['fake'])
label_df['real'] = fine_label(att_df['real'])
train_df['label'] = [get_label(row[TARGET]) for index, row in train_df.iterrows()]
train_df['event'] = [EVENT[row['category']] for index, row in train_df.iterrows()]
test_df['event'] = [EVENT[row['category']] for index, row in test_df.iterrows()]

# 分词
train_df['text'] = train_df.content.apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower()))) + \
                       train_df.comment_all.apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower())))
test_df['text'] = test_df.content.astype(str).apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower()))) + \
                      train_df.comment_all.apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower())))
label_df['text'] = label_df.content.astype(str).apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower()))) + \
                       label_df.comment_all.apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower())))



text_train = pd.DataFrame(train_df)
text_test = pd.DataFrame(test_df)

train_data, test_data, label_data, W = load_weight(args, text_train, text_test, label_df)


# 图片->训练数据
test_data = DatasetWithoutImg(label_data, mode='test_images')
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

# test
TextCNN.eval()
test_sub = np.zeros((len(label_df['id']), 3), dtype=np.float)
batch = len(label_df['id']) // args.batch_size

for i, (test_data, event_labels) in enumerate(test_loader):
    test_text, test_mask = to_var(
        test_data[0]), to_var(test_data[1])

    test_text = test_text.long()
    test_mask = test_mask.float()
    test_outputs, domain_outputs = TextCNN(test_text, test_mask)
    if i != batch:
        test_sub[i * args.batch_size:(i + 1) * args.batch_size, :] = to_np(test_outputs)
    else:
        test_sub[i * args.batch_size:len(test_df['id']), :] = to_np(test_outputs)

score_t = 0
for i in range(len(label_df)):
    score_t += np.dot(np.log(test_sub[i, :]), label_df.loc[i, TARGET].T) / len(label_df)
score = 1 / (1 - score_t)

test_pred = np.argmax(test_sub, axis=1)
test_true = [get_label(item[TARGET]) for item in label_df[TARGET].iterrows()]
test_accuracy = metrics.accuracy_score(test_true, test_pred)
test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
test_precision = metrics.precision_score(test_true, test_pred, average='macro')
test_recall = metrics.recall_score(test_true, test_pred, average='macro')



# 用训练集判断分数
'''score = 0
beta = [0.5, 0.3, 0.2]
for comment_num in range(COMMENT_NUMS+1):
    score_t = 0
    for i in range(2000):
        score_t += np.dot(np.log(test_train[i,comment_num*3:(comment_num+1)*3]), train_df.loc[i, TARGET].T) / 2000
    score_t += score_t * beta[comment_num]
score = 1 / (1 - score_t)
print('test on trainset: %f' % score)'''

# 输出结果

submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'fake_prob_label_0c': test_sub[:, 0],
    'fake_prob_label_2c': test_sub[:, 0],
    'fake_prob_label_all': test_sub[:, 0],
    'real_prob_label_0c': test_sub[:, 1],
    'real_prob_label_2c': test_sub[:, 1],
    'real_prob_label_all': test_sub[:, 1],
    'ncw_prob_label_0c': test_sub[:, 2],
    'ncw_prob_label_2c': test_sub[:, 2],
    'ncw_prob_label_all': test_sub[:, 2],
})

submission.to_csv('submission.csv', index=False)