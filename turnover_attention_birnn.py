
# coding: utf-8

# # 简介
# 
# 该代码基于English-French parallel corpus实现了机器翻译模型，模型在基础的Seq2Seq模型上加入Attention机制与BiRNN。代码采用Keras框架实现。

# # 1 - 加载包

# In[1]:

import warnings
warnings.filterwarnings("ignore")

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Reshape
from keras.layers import RepeatVector, Dense, Activation, Lambda, Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import keras
import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

# In[]:
def softmax(x, axis=1):
    """
    Softmax activation function.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


# In[18]:
# input len
Tx = 7
# 定义全局网络层对象
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor_tanh = Dense(32, activation="tanh")
densor_relu = Dense(1, activation="relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes=1)


# In[19]:

def one_step_attention(a, s_prev):
    """
    Attention机制的实现，返回加权后的Context Vector

    @param a: BiRNN的隐层状态
    @param s_prev: Decoder端LSTM的上一轮隐层输出

    Returns:
    context: 加权后的Context Vector
    """
    # 将s_prev复制Tx次
    s_prev = repeator(s_prev)
    # 拼接BiRNN隐层状态与s_prev
    concat = concatenator([a, s_prev])
    # 计算energies
    e = densor_tanh(concat)
    energies = densor_relu(e)
    # 计算weights
    alphas = activator(energies)
    # 加权得到Context Vector
    context = dotor([alphas, a])
    return context


# In[23]:
input_size = 1
output_size = 1

n_a = 32 # The hidden size of Bi-LSTM
n_s = 128 # The hidden size of LSTM in Decoder

decoder_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(output_size, activation=softmax)


# In[24]:

# 定义网络层对象（用在model函数中）
reshapor = Reshape((1, output_size))
concator = Concatenate(axis=-1)


# In[25]:

def define_model(Tx, Ty, n_a, n_s, source_vocab_size, target_vocab_size):
    """
    构造模型

    @param Tx: 输入序列的长度
    @param Ty: 输出序列的长度
    @param n_a: Encoder端Bi-LSTM隐层结点数
    @param n_s: Decoder端LSTM隐层结点数
    @param source_vocab_size: 输入（英文）语料的词典大小
    @param target_vocab_size: 输出（法语）语料的词典大小
    """

    # 定义输入层
    X = Input(shape=(Tx,))

    # Decoder端LSTM的初始状态
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')

    # Decoder端LSTM的初始输入
    out0 = Input(shape=(target_vocab_size, ), name='out0')
    out = reshapor(out0)

    s = s0
    c = c0

    # 模型输出列表，用来存储翻译的结果
    outputs = []

    # 定义Bi-LSTM
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    # Decoder端，迭代Ty轮，每轮生成一个翻译结果
    for t in range(Ty):

        # 获取Context Vector
        context = one_step_attention(a, s)

        # 将Context Vector与上一轮的翻译结果进行concat
        context = concator([context, reshapor(out)])
        s, _, c = decoder_LSTM_cell(context, initial_state=[s, c])

        # 将LSTM的输出结果与全连接层链接
        out = output_layer(s)

        # 存储输出结果
        outputs.append(out)

    model = Model([X, s0, c0, out0], outputs)

    return model


# In[29]:

# 初始化各类向量
m = X.shape[0]
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
out0 = np.zeros((m, output_size))
outputs = list(Y.swapaxes(0, 1))


# In[ ]:

is_train = False
load_model = True
model = define_model(Tx, Ty, n_a, n_s, len(source_vocab_to_int), len(target_vocab_to_int))
model.summary()
model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001),
              metrics=['accuracy'],
              loss='categorical_crossentropy')

# In[ ]:
file_path = r"D:\MyProject\mt_attention_birnn\pretrained_seq2seq_model.h5"
if load_model:
    # 加载结构
    # model = model_from_json(open('my_model_architecture.json').read())
    # 加载参数
    model.load_weights(file_path)



if is_train:
    # 训练模型
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1
                                 , save_best_only=True, save_weights_only=True, mode='min')
    model.fit([X, s0, c0, out0], outputs, epochs=5, batch_size=128, validation_split=0.2
              , shuffle=True, verbose=2, callbacks=[checkpoint])
    # 保存结构
    # json_string = model.to_json()
    # open('my_model_architecture.json', 'w').write(json_string)
    # 保存参数
    # model.save_weights(file_path)




# ## 3.3 预测
# In[ ]:

def make_prediction(sentence):
    """
    对给定的句子进行翻译
    """
    # 将句子分词后转化为数字编码
    unk_idx = source_vocab_to_int["<UNK>"]
    word_idx = [source_vocab_to_int.get(word, unk_idx) for word in sentence.lower().split()]
    
    word_idx = np.array(word_idx + [0] * (20 - len(word_idx)))
    
    # 翻译结果
    preds = model.predict([word_idx.reshape(-1,20), s0, c0, out0])
    predictions = np.argmax(preds, axis=-1)
    
    # 转换为单词
    idx = [target_int_to_vocab.get(idx[0], "<UNK>") for idx in predictions]
    
    # 返回句子
    return " ".join(idx)


# In[ ]:
while(1):
    your_sentence = input("Please input your sentences: ")
    if your_sentence != 'end':
        print(make_prediction(your_sentence))
    else:
        break



