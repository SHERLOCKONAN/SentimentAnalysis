# -*- coding: utf-8 -*-
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import  load_model
from keras.layers import Embedding
import keras.callbacks
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import os
import csv


import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#限制最多GPU占用为30%
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


train_content_list =[]
train_label_list =[]
test_id_list = []
test_content_list =[]
def readTrainData():
    #使用pandas读取训练集
    dataSetPath = "labeledTrainData.csv"
    train = pd.read_csv(dataSetPath,encoding="ISO-8859-1")
    #将读取出来的数据转为DataFram结构
    df = pd.DataFrame(train)
    #计算数据的有多少行
    lineCount = df.size//3
    for i in range (0,lineCount):
        label = df.iloc[i][1]#label
        review = df.iloc[i][2]#review
        train_content_list.append(review)
        train_label_list.append(label)
def readTestData():
    #使用pandas读取测试集
    dataSetPath = "testData.csv"
    train = pd.read_csv(dataSetPath,encoding="ISO-8859-1")
    #将读取出来的数据转为DataFram结构
    df = pd.DataFrame(train)
    #计算数据的有多少行
    lineCount = df.size//2

    for i in range (0,lineCount):
        id = df.iloc[i][0]#id
        review = df.iloc[i][1]#review
        test_content_list.append(review)
        test_id_list.append(id)

def parseStopWord():
    #去掉停用词
    stop_words=set(stopwords.words('english'))
    for con in content_list:
        words = nltk.word_tokenize(con)
        line = ""
        for word in words:
            if word.isalpha() and word not in stop_words:
                line = line+word+" "
        seq.append(line)




def buildword2vec():
    #构建单词->词向量词典
    f = open(os.path.join("./Word2Vec/300features_10minwords_10context.txt"))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

def buildVecMatrix():
    #构建词向量矩阵
    embedding_matrix = np.zeros((len(word_index) + 1, 300),dtype='float32')
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector





# 训练结合初始化
content_list = []
seq = []
embeddings_index = {}

readTrainData()
readTestData()
print("Read Data Finished!")
content_list = train_content_list + test_content_list
parseStopWord()

#序列化处理
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(content_list)
sequences = tokenizer.texts_to_sequences(content_list)
word_index = tokenizer.word_index
final_sequences=sequence.pad_sequences(sequences,maxlen=500)

train_final_seqs = final_sequences[0:len(train_content_list)]
train_final_seqs_len = len(train_final_seqs)
test_final_seqs = final_sequences[len(train_content_list):len(final_sequences)]
test_final_seqs_len = len(test_final_seqs)

buildword2vec()
# 构建词向量矩阵
embedding_matrix = np.zeros((len(word_index) + 1, 300),dtype="float32")

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


label=np.array(train_label_list).astype(int)
X=train_final_seqs
y=label


# 划分测试集和训练集
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2)
#
#网络构建

#EarlyStop
earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02,patience=0, verbose=0, mode='auto')
callbackList=[]
callbackList.append(earlyStop)

embedding_layer = Embedding(len(word_index) + 1,
                            300,
                            weights=[embedding_matrix],
                            input_length=500,
                            trainable=False)

model=Sequential()
model.add(embedding_layer)
model.add(LSTM(128,dropout=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(Xtrain,ytrain,batch_size=64,epochs=20,validation_data=(Xtest,ytest),callbacks=callbackList)
model.save("lstm_model.h5")
print("save model succeed~")


print("prepare to validate")
#save model
model = load_model("lstm_model.h5")

result = model.predict_classes(test_final_seqs)

#设置结果输出文件位置
resultPath =  "LSTMresult.csv"
csv_file = open(resultPath,"w+")
csv_writer = csv.writer(csv_file)
#文件头
head = ("id","sentiment")
csv_writer.writerow(head)

for i in range (0,test_final_seqs_len):
    prediction = (test_id_list[i],result[i][0])
    csv_writer.writerow(prediction)

csv_file.close()











