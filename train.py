import jieba
import torch
import argparse
import time
import ipdb
import pickle
import pandas as pd
import torch.nn as nn
import numpy as np

import os
from model import CNN_Fusion, to_var, to_np
from processing import *
from dataset import DatasetWithImg, DatasetWithoutImg
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 默认参数(可以做修改选取效果好的参数)
COMMENT_NUMS = 2
maxlen = 80

# 超参
parser = argparse.ArgumentParser()

parser.add_argument('--vocab_size', type=int, default=300, help='vocabulary size')
parser.add_argument('--crop_size', type=int, default=112, help='crop size')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dimension')
parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--embedding_dim', type=int, default=32, help='embedding dimension')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.005, help='learning rate')
parser.add_argument('--lambd', type=int, default=1, help='lambda')
parser.add_argument('--dropout', type=int, default=0.5, help='dropout')
parser.add_argument('--event_num', type=int, default=9, help='event number')
args = parser.parse_args()

# 清洗数据
train_df = pd.read_csv('data/train.csv',
                       skiprows=[0],
                       dtype={'picture_lists':str},
                       names=['id', 'content', 'picture_lists', 'category', 'ncw', 'fake', 'real','comment_2', 'comment_all']
                       )
test_df = pd.read_csv('data/test_dataset.csv',
                      skiprows=[0],
                      dtype={'content':str, 'picture_lists':str},
                      names=['id', 'content', 'picture_lists', 'category', 'comment_2', 'comment_all']
                      )
train_df['label'] = [get_label(row[TARGET]) for index, row in train_df.iterrows()]

# 分词
train_df['text_cut'] = train_df.content.apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower()))) + \
                       train_df.comment_all.apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower())))
test_df['text_cut'] = test_df.content.astype(str).apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower()))) + \
                      train_df.comment_all.apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower())))

# 整理数据
train_data, text_train = divide_data(train_df, mode='train_pictures')
test_data, text_test = divide_data(test_df, mode='test_images')
train_data, validation_data = split_train_validation(train_data, 0.95, text_only=False)
text_train, validation_text = split_train_validation(text_train, 0.95, text_only=True)

train_data = pd.DataFrame(train_data)
text_train = pd.DataFrame(text_train)
test_data = pd.DataFrame(test_data)
text_test = pd.DataFrame(text_test)
validation_data = pd.DataFrame(validation_data)
validation_text = pd.DataFrame(validation_text)

train_data, test_data, validation_data, W = load_weight(args, train_data, test_data, validation_data)

# 图片->训练数据
train_data = DatasetWithImg(train_data, crop_size=args.crop_size, mode='train_pictures')
test_data = DatasetWithImg(test_data, crop_size=args.crop_size, mode='test_images')
validation_data = DatasetWithImg(validation_data, crop_size=args.crop_size, mode='train_pictures')
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
validation_loader = DataLoader(dataset=validation_data, batch_size=args.batch_size, shuffle=False)


print('building model')
model = CNN_Fusion(args, W)

if torch.cuda.is_available():
    print("USE CUDA")
    model.cuda()

# 训练
test_sub = np.zeros((len(test_df), 9), dtype=np.float)
test_train = np.zeros((2000, 9), dtype=np.float)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                             lr=args.learning_rate)

iter_per_epoch = len(train_data)
print("loader size %d  %d" % (len(train_data), len(text_train)))
best_validate_acc = 0.000
best_test_acc = 0.000
best_loss = 100
best_validate_dir = ''
best_list = [0, 0]

for epoch in range(args.num_epochs):
    p = float(epoch) / 100
    # lambd = 2. / (1. + np.exp(-10. * p)) - 1
    lr = 0.001 / (1. + 10 * p) ** 0.75

    optimizer.lr = lr
    # rgs.lambd = lambd
    start_time = time.time()
    cost_vector = []
    class_cost_vector = []
    domain_cost_vector = []
    acc_vector = []
    valid_acc_vector = []
    test_acc_vector = []
    vali_cost_vector = []
    test_cost_vector = []
    # if i == 0:
    #     train_score = to_np(class_outputs.squeeze())
    #     train_pred = to_np(argmax.squeeze())
    #     train_true = to_np(train_labels.squeeze())
    # else:
    #     class_score = np.concatenate((train_score, to_np(class_outputs.squeeze())), axis=0)
    #     train_pred = np.concatenate((train_pred, to_np(argmax.squeeze())), axis=0)
    #     train_true = np.concatenate((train_true, to_np(train_labels.squeeze())), axis=0)

    for i, (train_data, train_labels, event_labels) in enumerate(train_loader):
        train_text, train_image, train_mask, train_labels, event_labels = \
            to_var(train_data[0]), to_var(train_data[1]), to_var(train_data[2]), \
            to_var(train_labels), to_var(event_labels)

        train_text = train_text.long()
        train_image = train_image.float()
        train_mask = train_mask.float()
        # Forward + Backward + Optimize
        optimizer.zero_grad()

        class_outputs, domain_outputs = model(train_text, train_image, train_mask)

        ## Fake or Real loss
        class_loss = criterion(class_outputs, train_labels)
        # Event Loss
        domain_loss = criterion(domain_outputs, event_labels)
        loss = class_loss + domain_loss
        loss.backward()
        optimizer.step()
        _, argmax = torch.max(class_outputs, 1)

        cross_entropy = True

        if True:
            accuracy = (train_labels == argmax.squeeze()).float().mean()
        else:
            _, labels = torch.max(train_labels, 1)
            accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()

        class_cost_vector.append(class_loss.data[0])
        domain_cost_vector.append(domain_loss.data[0])
        cost_vector.append(loss.data[0])
        acc_vector.append(accuracy.data[0])
        # if i == 0:
        #     train_score = to_np(class_outputs.squeeze())
        #     train_pred = to_np(argmax.squeeze())
        #     train_true = to_np(train_labels.squeeze())
        # else:
        #     class_score = np.concatenate((train_score, to_np(class_outputs.squeeze())), axis=0)
        #     train_pred = np.concatenate((train_pred, to_np(argmax.squeeze())), axis=0)
        #     train_true = np.concatenate((train_true, to_np(train_labels.squeeze())), axis=0)

    model.eval()
    validate_acc_vector_temp = []
    for i, (validate_data, validate_labels, event_labels) in enumerate(validation_loader):
        validate_text, validate_image, validate_mask, validate_labels, event_labels = \
            to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), \
            to_var(validate_labels), to_var(event_labels)
        validate_text = validate_text.long()
        validate_image = validate_image.float()
        validate_mask = validate_mask.float()

        validate_outputs, domain_outputs = model(validate_text, validate_image, validate_mask)
        _, validate_argmax = torch.max(validate_outputs, 1)
        vali_loss = criterion(validate_outputs, validate_labels)
        # domain_loss = criterion(domain_outputs, event_labels)
        # _, labels = torch.max(validate_labels, 1)

        validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
        vali_cost_vector.append(vali_loss.data[0])
        # validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
        validate_acc_vector_temp.append(validate_accuracy.data[0])
    validate_acc = np.mean(validate_acc_vector_temp)
    valid_acc_vector.append(validate_acc)
    model.train()
    print('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
          % (
              epoch + 1, args.num_epochs, np.mean(cost_vector), np.mean(class_cost_vector), np.mean(domain_cost_vector),
              np.mean(acc_vector), validate_acc))

    if validate_acc > best_validate_acc:
        best_validate_acc = validate_acc

        best_validate_dir = 'checkpoint/_WithImage' + str(epoch + 1) + '.pkl'
        torch.save(model.state_dict(), best_validate_dir)

    duration = time.time() - start_time
    # print ('Epoch: %d, Mean_Cost: %.4f, Duration: %.4f, Mean_Train_Acc: %.4f, Mean_Test_Acc: %.4f'
    # % (epoch + 1, np.mean(cost_vector), duration, np.mean(acc_vector), np.mean(test_acc_vector)))
    # best_validate_dir = args.output_file + 'weibo_GPU2_out.' + str(52) + '.pkl'

# Test the Model
print('testing model')
model = CNN_Fusion(args, W)
model.load_state_dict(torch.load(best_validate_dir))
#    print(torch.cuda.is_available())
if torch.cuda.is_available():
    model.cuda()
model.eval()
test_score = []
test_pred = []
test_true = []
for i, (test_data, event_labels) in enumerate(test_loader):
    test_text, test_image, test_mask = to_var(
        test_data[0]), to_var(test_data[1]), to_var(test_data[2])

    test_text = test_text.long()
    test_image = test_image.float()
    test_mask = test_mask.float()
    test_outputs, domain_outputs = model(test_text, test_image, test_mask)
    _, test_argmax = torch.max(test_outputs, 1)

print(test_outputs)
print(test_argmax)

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
    'fake_prob_label_2c': test_sub[:, 3],
    'fake_prob_label_all': test_sub[:, 6],
    'real_prob_label_0c': test_sub[:, 1],
    'real_prob_label_2c': test_sub[:, 4],
    'real_prob_label_all': test_sub[:, 7],
    'ncw_prob_label_0c': test_sub[:, 2],
    'ncw_prob_label_2c': test_sub[:, 5],
    'ncw_prob_label_all': test_sub[:, 8],
})

submission.to_csv('submission.csv', index=False)