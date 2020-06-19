import jieba
import torch
import argparse
import logging
import time
import ipdb
import pickle
import pandas as pd
import torch.nn as nn
import numpy as np

from sklearn import metrics
from model import TextCNN, to_var, to_np
from processing import *
from dataset import DatasetWithoutImg
from torch.utils.data import DataLoader

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 超参
parser = argparse.ArgumentParser()

parser.add_argument('--vocab_size', type=int, default=300, help='vocabulary size')
parser.add_argument('--crop_size', type=int, default=112, help='crop size')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dimension')
parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--embedding_dim', type=int, default=32, help='embedding dimension')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--lambd', type=int, default=1, help='lambda')
parser.add_argument('--dropout', type=int, default=0.5, help='dropout')
parser.add_argument('--event_num', type=int, default=2, help='event number')
args = parser.parse_args()

# 清洗数据
train_df = pd.read_csv('data/train.csv',
                       skiprows=[0],
                       names=['id', 'content', 'picture_lists', 'category', 'ncw', 'fake', 'real', 'comment_2',
                              'comment_all']
                       )
test_df = pd.read_csv('data/test_dataset.csv',
                      skiprows=[0],
                      dtype={'comment': str},
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
label_df['ncw'] = fine_label(att_df['ncw'])
label_df['fake'] = fine_label(att_df['fake'])
label_df['real'] = fine_label(att_df['real'])

train_df = train_df[train_df['category'] == '科技']
label_df['event'] = 1

test_df.reset_index(drop=True, inplace=True)
label_df.reset_index(drop=True, inplace=True)

# 分词
train_df['text_cut'] = train_df.content.apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower()))) + \
                       train_df.comment_all.apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower())))
test_df['text_cut'] = test_df.content.astype(str).apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower()))) + \
                      test_df.comment_all.apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower())))
label_df['text'] = label_df.content.astype(str).apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower()))) + \
                       label_df.comment_all.apply(lambda x: ' '.join(jieba.cut(cut_sub(x).lower())))

train_data, text_train = divide_data(train_df, mode='train_pictures')
test_data, text_test = divide_data(test_df, mode='test_images')

text_train = pd.DataFrame(text_train)
text_test = pd.DataFrame(text_test)

train_data, test_data, label_data, W = load_weight(args, text_train, text_test, label_df)

# 图片->训练数据
train_data = DatasetWithoutImg(train_data, mode='train_pictures')
test_data = DatasetWithoutImg(test_data, mode='test_images')
label_data = DatasetWithoutImg(label_data, mode='test_images')

source_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
domain_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(dataset=label_data, batch_size=args.batch_size, shuffle=False)

print('building model')
model = TextCNN(args, W)

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
    len_dataloader = min(len(source_loader), len(domain_loader))

    data_source_iter = iter(source_loader)
    data_target_iter = iter(domain_loader)

    i = 0

    p = float(epoch) / args.num_epochs
    lr = 0.001 / (1. + 10 * p) ** 0.75
    optimizer.lr = lr

    while(i < len_dataloader):

        # training model using source data
        data_source = data_source_iter.next()
        train_data, train_labels, event_labels = data_source[0], to_var(data_source[1]), to_var(data_source[2])
        train_text, train_mask = to_var(train_data[0]), to_var(train_data[1])

        train_text = train_text.long()
        train_mask = train_mask.float()

        optimizer.zero_grad()

        class_outputs, domain_outputs = model(train_text, train_mask)

        class_loss = criterion(class_outputs, train_labels)
        domain_s_loss = criterion(domain_outputs, event_labels)

        # training model using target data
        data_target = data_target_iter.next()
        train_data, train_labels, event_labels = data_source[0], to_var(data_source[1]), to_var(data_source[2])
        train_text, train_mask = to_var(train_data[0]), to_var(train_data[1])

        train_text = train_text.long()
        train_mask = train_mask.float()

        _, domain_outputs = model(train_text, train_mask)

        domain_t_loss = criterion(domain_outputs, event_labels)
        err = class_loss + domain_s_loss + domain_t_loss
        err.backward()
        optimizer.step()

        i += 1

        print('epoch: %d, [iter: %d / all %d], class_loss: %f, domain_s_loss: %f, domain_t_loss: %f' \
        % (epoch, i, len_dataloader, class_loss.cpu().data.numpy(),
           domain_s_loss.cpu().data.numpy(), domain_t_loss.cpu().data.numpy()))
        logging.info('epoch: %d, [iter: %d / all %d], class_loss: %f, domain_s_loss: %f, domain_t_loss: %f' \
        % (epoch, i, len_dataloader, class_loss.cpu().data.numpy(),
           domain_s_loss.cpu().data.numpy(), domain_t_loss.cpu().data.numpy()))

        dir = 'checkpoint/WithoutImage_' + str(epoch + 1) + '.pkl'
        torch.save(model.state_dict(), dir)

# test
model = TextCNN(args, W)
model.load_state_dict(torch.load(dir))
if torch.cuda.is_available():
    model.cuda()
model.eval()
test_sub = np.zeros((len(label_df['id']), 3), dtype=np.float)
batch = len(label_df['id']) // args.batch_size

for i, (test_data, event_labels) in enumerate(test_loader):
    test_text, test_mask = to_var(
        test_data[0]), to_var(test_data[1])

    test_text = test_text.long()
    test_mask = test_mask.float()
    test_outputs, domain_outputs = model(test_text, test_mask)
    if i != batch:
        test_sub[i * args.batch_size:(i + 1) * args.batch_size, :] = to_np(test_outputs)
    else:
        test_sub[i * args.batch_size:len(test_df['id']), :] = to_np(test_outputs)

score_t = 0
for i in range(len(label_df)):
    score_t += np.dot(np.log(test_sub[i, :]), label_df.loc[i, TARGET].T) / len(label_df)
score = 1 / (1 - score_t)
print(score)

test_pred = np.argmax(test_sub, axis=1)
test_true = [get_label(item[TARGET]) for index, item in label_df.iterrows()]
test_accuracy = metrics.accuracy_score(test_true, test_pred)
test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
test_precision = metrics.precision_score(test_true, test_pred, average='macro')
test_recall = metrics.recall_score(test_true, test_pred, average='macro')

test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)
print("Classification Acc: %.4f, Precision: %.4f, Recall: %.4f, f1 Score: %.4f"
      % (test_accuracy, test_precision, test_recall, test_f1))
print("Classification report:\n%s\n"
      % (metrics.classification_report(test_true, test_pred)))
print("Classification confusion matrix:\n%s\n"
      % (test_confusion_matrix))

'''p = float(epoch) / 100
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
    train_text, train_mask, train_labels, event_labels = \
        to_var(train_data[0]), to_var(train_data[1]), \
        to_var(train_labels), to_var(event_labels)

    train_text = train_text.long()
    train_mask = train_mask.float()
    # Forward + Backward + Optimize
    optimizer.zero_grad()

    class_outputs, domain_outputs = model(train_text, train_mask)

    ## Fake or Real loss
    class_loss = criterion(class_outputs, train_labels)
    # Event Loss
    domain_loss = criterion(domain_outputs, event_labels)
    # print(domain_outputs, train_labels)
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
    validate_text, validate_mask, validate_labels, event_labels = \
        to_var(validate_data[0]), to_var(validate_data[1]), \
        to_var(validate_labels), to_var(event_labels)
    validate_text = validate_text.long()
    validate_mask = validate_mask.float()

    validate_outputs, domain_outputs = model(validate_text, validate_mask)
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

    best_validate_dir = 'checkpoint/WithoutImage_' + str(epoch + 1) + '.pkl'
    torch.save(model.state_dict(), best_validate_dir)

duration = time.time() - start_time
# print ('Epoch: %d, Mean_Cost: %.4f, Duration: %.4f, Mean_Train_Acc: %.4f, Mean_Test_Acc: %.4f'
# % (epoch + 1, np.mean(cost_vector), duration, np.mean(acc_vector), np.mean(test_acc_vector)))
# best_validate_dir = args.output_file + 'weibo_GPU2_out.' + str(52) + '.pkl'''''
