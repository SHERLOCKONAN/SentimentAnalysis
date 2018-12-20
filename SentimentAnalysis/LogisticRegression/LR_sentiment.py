# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import os
import csv
import time
import itertools

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



# 训练结合初始化
content_list = []
seq = []
embeddings_index = {}

train_content_list =[]
train_label_list =[]
test_id_list = []
test_content_list =[]
def readTrainData():
    #使用pandas读取训练集
    dataSetPath = "../labeledTrainData.csv"
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
    dataSetPath = "../testData.csv"
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
    f = open(os.path.join("../Word2Vec/300features_10minwords_10context.txt"))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

def makeFeatureVec(words,num_features):
    # 用于平均给定段落中的所有单词向量的函数
    #
    # 预初始化一个空的 numpy 数组（为了速度）
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word 是一个列表，包含模型词汇表中的单词名称。
    # 为了获得速度，将其转换为集合。
    #index2word_set = set(model.index2word)
    #
    # 遍历评论中的每个单词，如果它在模型的词汇表中，
    # 则将其特征向量加到 total
    for word in words:
        if embeddings_index.get(word) is not None:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,embeddings_index.get(word))
    featureVec = featureVec/nwords
    return featureVec





print("Read Data begin!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
readTrainData()
readTestData()
print("Read Data Finished!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
content_list = train_content_list + test_content_list
parseStopWord()
print("buildword2vec begin!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
buildword2vec()
print("buildword2vec Finished!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))

print("Cal Mean Vec begin!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
#计算每个句子的平均词向量
#model = Word2Vec.load("vectors.txt")
reviewFeatureVecs = np.zeros((len(seq), 300), dtype="float32")
counter = 0
for line in seq:
    list = line.split(" ")
    reviewFeatureVecs[counter]=makeFeatureVec(list,300)
    counter=counter+1
print("Cal Mean Vec end!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))

# pca = decomposition.PCA(n_components=300)
# reduced_Feature = pca.fit_transform(reviewFeatureVecs)
# print("reduced dimesion end!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))

reduced_Feature = reviewFeatureVecs
#numpy数组截断不生成新的数组，只返回原数组的引用
train_final_seqs = reduced_Feature[0:len(train_content_list)]
train_final_seqs_len = len(train_final_seqs)
test_final_seqs = reduced_Feature[len(train_content_list):len(reduced_Feature)]
test_final_seqs_len = len(test_final_seqs)

LR_model = LogisticRegression()
LR_model = LR_model.fit(train_final_seqs, train_label_list)
predicted = LR_model.predict(test_final_seqs)

print("Pretiction end!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))

#设置结果输出文件位置
resultPath =  "SVMresult.csv"
csv_file = open(resultPath,"w+")
csv_writer = csv.writer(csv_file)
#文件头
head = ("id","sentiment")
csv_writer.writerow(head)

for i in range (0,test_final_seqs_len):
    prediction = (test_id_list[i],predicted[i])
    csv_writer.writerow(prediction)

csv_file.close()



