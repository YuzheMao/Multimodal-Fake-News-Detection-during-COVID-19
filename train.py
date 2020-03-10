import jieba
import pandas as pd
import numpy as np
from math import log
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras import layers

# 默认参数(可以做修改选取效果好的参数)
COMMENT_NUMS = 2
BATCH_SIZE = 512
EPOCHS = 4
NUM_MODELS = 2
TARGET = {'fake', 'real', 'ncw'}
maxlen = 30
embedding_dim = 50


# 清洗数据
train_df = pd.read_csv('data/train.csv',
                       skiprows=[0],
                       names=['id', 'comment', 'picture_lists', 'category', 'ncw', 'fake', 'real','comment_2', 'comment_all']
                       )
test_df = pd.read_csv('data/test_dataset.csv',
                      skiprows=[0],
                      dtype={'comment':str},
                      names=['id', 'comment', 'picture_lists', 'category', 'comment_2', 'comment_all']
                      )

# 分词
train_df['text_cut'] = train_df.comment.apply(lambda x: ' '.join(jieba.cut(x)))
test_df['text_cut'] = test_df.comment.astype(str).apply(lambda x: ' '.join(jieba.cut(x)))

# 转换词向量
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(train_df['text_cut']) + list(test_df['text_cut']))
train_x = tokenizer.texts_to_sequences(train_df['text_cut'])
test_x = tokenizer.texts_to_sequences(test_df['text_cut'])

vocab_size = len(tokenizer.word_index) +1

train_x = pad_sequences(train_x, padding='post', maxlen=maxlen)
train_y = train_df[TARGET].values
aux_train_y = train_df['category'].values
test_x = pad_sequences(test_x, padding='post', maxlen=maxlen)

# 搭模型
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                          output_dim=embedding_dim,
                          input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])


# 训练
test_sub = np.zeros((len(test_x), 9), dtype=np.float)
test_train = np.zeros((2000, 9), dtype=np.float)
for i in range(COMMENT_NUMS+1):
    model.summary()
    model.fit(train_x[:300], train_y[:300],
             epochs=1,
             batch_size=BATCH_SIZE
              )

    test_sub[:, i*3:(i+1)*3] = model.predict(test_x)
    test_train[:, i*3:(i+1)*3] = model.predict(train_x[:2000])
print(test_sub)


# 用训练集判断分数
score = 0
beta = [0.5, 0.3, 0.2]
for comment_num in range(COMMENT_NUMS+1):
    score_t = 0
    for i in range(1999):
        score_t += np.dot(np.log(test_train[i,comment_num*3:(comment_num+1)*3]), train_df.loc[0, TARGET].T) / 2000
    score_t += score_t * beta[comment_num]
score = 1 / (1 - score_t)
print('test on trainset: %f' % score)

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
