import os
import cv2
import math
import random
import pickle
import copy
import numpy as np
from PIL import Image
from torchtext import data
from tqdm import tqdm
from random import sample

load_img = lambda img_path: np.array(Image.open(img_path))
TARGET = {'fake': 0,
          'real': 1,
          'ncw': 2}
EVENT = {'疫情': 0,
         '科技': 1,
         '政治': 2,
         '军事': 3,
         '财经商业': 4,
         '社会生活': 5,
         '文体娱乐': 6,
         '医药健康': 7,
         '教育考试': 8}

# 只有2条的就去\t
def cut_sub(str):
    if type(str) == float:
        return ' '
    return str.replace('\t', '')

# 以\t切割图片名
def cut_imgs(imgs):
    return imgs.split('\t')

# 读取图片
def load_img_fast_jpg(img_path):
    x = cv2.imread(img_path, cv2.COLOR_GRAY2RGB)
    return x


# 改one-bot标签为正常标签
def get_label(target):
    for index, col in target.iteritems():
        if col == 1:
            return TARGET[index]

def fine_label(labels):
    return [1 if item == 0.5 else 0 for item in labels]


# 剪裁图片至统一大小()
def get_crop(img, crop_size, random_crop=False):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    half_crop = crop_size // 2
    pad_x = max(0, crop_size - img.shape[1])
    pad_y = max(0, crop_size - img.shape[0])
    if (pad_x > 0) or (pad_y > 0):
        img = np.pad(img, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2), (0,0)), mode='wrap')
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    if random_crop:
        freedom_x, freedom_y = img.shape[1] - crop_size, img.shape[0] - crop_size
        if freedom_x > 0:
            center_x += np.random.randint(math.ceil(-freedom_x / 2), freedom_x - math.floor(freedom_x / 2))
        if freedom_y > 0:
            center_y += np.random.randint(math.ceil(-freedom_y / 2), freedom_y - math.floor(freedom_y / 2))

    return img[center_y - half_crop: center_y + crop_size - half_crop,
           center_x - half_crop: center_x + crop_size - half_crop]

# 一张图片切割成64块分别DCT再拼合起来
def DCT_block(img):
    dis_num = 8
    dis = 28
    dct_img = np.zeros(img.shape)
    hists = np.zeros((250, dis_num*dis_num))
    for i in range(dis_num):
        for j in range(dis_num):
            dct_img[dis*i:dis*(i+1), dis*j:dis*(j+1), 0] = cv2.dct(np.float32(img[dis*i:dis*(i+1), dis*j:dis*(j+1), 0]))
            dct_img[dis*i:dis*(i+1), dis*j:dis*(j+1), 1] = cv2.dct(np.float32(img[dis*i:dis*(i+1), dis*j:dis*(j+1), 1]))
            dct_img[dis*i:dis*(i+1), dis*j:dis*(j+1), 2] = cv2.dct(np.float32(img[dis*i:dis*(i+1), dis*j:dis*(j+1), 2]))
            hist, bin_edges = np.histogram(dct_img[dis*i:dis*(i+1), dis*j:dis*(j+1)], 250, [0, 255])
            hists[:, i*dis_num+j] = hist
    return hists

def img_identify(path):
    x = load_img_fast_jpg(path)
    if x is None:
        return False
    elif x.ndim != 3:
        return False
    elif x.shape[2] != 3:
        return False
    return True



# 把图片和标签一一对应
def divide_data(df, mode):
    img_list = []
    label_list = []
    id_list = []
    text_list = []
    event_list = []

    label_list2 = []
    id_list2 = []
    text_list2 = []
    event_list2 = []
    for index, item in df.iterrows():
        if not type(item['picture_lists']) == float:
            imgs = cut_imgs(item['picture_lists'])
            # 全取
            for img in imgs:
                if img_identify(os.path.join('data', mode, img)):
                    img_list.append(img)
                    id_list.append(item['id'])
                    text_list.append(item['text_cut'])
                    event_list.append(EVENT[item['category']])
                    if mode == 'train_pictures':
                        label_list.append(get_label(item[TARGET]))
                else:
                    if mode == 'train_pictures':
                        label_list2.append(get_label(item[TARGET]))
                    id_list2.append(item['id'])
                    text_list2.append(item['text_cut'])
                    event_list2.append(EVENT[item['category']])
        else:
            if mode == 'train_pictures':
                label_list2.append(get_label(item[TARGET]))
            id_list2.append(item['id'])
            text_list2.append(item['text_cut'])
            event_list2.append(EVENT[item['category']])
        # 随机取一个
        # img = imgs[random.randint(0, len(imgs) - 1)] if len(imgs) > 1 else imgs[0]
        # img_list.append(img)
        # id_list.append(item['id'])
        # text_list.append(item['text_cut'])
        # event_list.append(EVENT[item['category']])
        # if mode == 'train_pictures':
        #     label_list.append(get_label(item[TARGET]))

    if mode == 'train_pictures':
        return {'id':id_list, 'image': img_list, 'text': text_list, 'event': event_list, 'label': label_list},\
               {'id':id_list2, 'text': text_list2, 'event': event_list2, 'label': label_list2}
    elif mode == 'test_images':
        return {'id':id_list, 'image': img_list, 'text': text_list, 'event': event_list},\
               {'id': id_list2, 'text': text_list2, 'event': event_list2}

# 处理图像
def process_img(mode, img, crop_size):
    img = load_img_fast_jpg(os.path.join('data', mode, img))
    img = get_crop(img, crop_size * 2)
    # img = DCT_block(img)
    img = np.transpose(img, (2, 0, 1))
    return img

def select(train, selec_indices, text_only):
    img_list = []
    label_list = []
    id_list = []
    text_list = []
    event_list = []
    for i in selec_indices:
        if not text_only:
            img_list.append(train['image'][i])
        label_list.append(train['label'][i])
        id_list.append(train['id'][i])
        text_list.append(train['text'][i])
        event_list.append(train['event'][i])
    if not text_only:
        return {'id': id_list, 'image': img_list, 'text': text_list, 'event': event_list, 'label': label_list}
    else:
        return {'id': id_list, 'text': text_list, 'event': event_list, 'label': label_list}


def add_unknown_words(word_vecs, vocab, min_df=1, k=32):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

def word2vec(post, word_id_map, W, args):
    word_embedding = []
    mask = []
    #length = []

    for sentence in post:
        sen_embedding = []
        seq_len = len(sentence) -1
        mask_seq = np.zeros(args.sequence_len, dtype = np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence.split()):
            if word in word_id_map:
                sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)


        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
        #length.append(seq_len)
    return word_embedding, mask

def load_weight(args, train_data, test_data, label_data):
    word_vector_path = 'data/word_embedding.pickle'
    f = open(word_vector_path, 'rb')
    weight = pickle.load(f)  # W, W2, word_idx_map, vocab
    W, W2, word_idx_map, vocab, max_len = weight[0], weight[1], weight[2], weight[3], weight[4]

    # add_unknown_words(weight, vocab)
    args.vocab_size = len(vocab)
    args.sequence_len = max_len
    print("translate data to embedding")

    print("translate test data to embedding")
    word_embedding, mask = word2vec(label_data['text'], word_idx_map, W, args)
    label_data['text'] = word_embedding
    label_data['mask'] = mask
    word_embedding, mask = word2vec(test_data['text'], word_idx_map, W, args)
    test_data['text'] = word_embedding
    test_data['mask'] = mask
    #test[-2]= transform(test[-2])
    word_embedding, mask = word2vec(train_data['text'], word_idx_map, W, args)
    train_data['text'] = word_embedding
    train_data['mask'] = mask
    print("sequence length " + str(args.sequence_len))
    print("Train Data Size is "+str(len(train_data['text'])))
    print("Finished loading data ")
    return train_data, test_data, label_data, W

def split_train_validation(train, percent, text_only):
    whole_len = len(train['id'])

    train_indices = (sample(range(whole_len), int(whole_len * percent)))
    train_data = select(train, train_indices, text_only)
    print("train data size is "+ str(len(train_data['id'])))
    # print()

    validation = select(train, np.delete(range(whole_len), train_indices), text_only)
    print("validation size is "+ str(len(validation['id'])))
    print("train and validation data set has been splited")

    return train_data, validation