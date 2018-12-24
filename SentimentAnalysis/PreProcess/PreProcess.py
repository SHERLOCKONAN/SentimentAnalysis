# -*- coding: utf-8 -*-
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import os
import csv
import time
import itertools
import pickle
from gensim.models import word2vec

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

if_lower = True
if if_lower:
    lower = "lower_"
else:
    lower = ""

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
def readUnlabelTrainData():
    #使用pandas读取测试集
    dataSetPath = "../unlabeledTrainData.csv"
    train = pd.read_csv(dataSetPath,encoding="ISO-8859-1")
    #将读取出来的数据转为DataFram结构
    df = pd.DataFrame(train)
    #计算数据的有多少行
    lineCount = df.size//2
    for i in range (0,lineCount):
        try:
            if(len(df.iloc[i])==2):
                review = df.iloc[i][1]#review
                train_content_list.append(review)
        except:
            print(i)
        else:
            continue
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

# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def parseStopWord():
    #去掉停用词 + 词性还原
    stop_words=set(stopwords.words('english'))
    wnl = WordNetLemmatizer()

    for con in content_list:
        words = nltk.word_tokenize(con)
        tagged_sent = pos_tag(words)     # 获取单词词性
        line = ""
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            # 词形还原
            new_word = wnl.lemmatize(tag[0], pos=wordnet_pos)
            if new_word.isalpha() and new_word not in stop_words:
            #if new_word.isalpha():
                if if_lower:
                    new_word = new_word.lower()
                line = line+new_word+" "
        seq.append(line)

def buildword2vec():
    f = open(os.path.join("./"+lower+"300features_5minwords_10context.txt"))
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

readTrainData()
label_train_len = len(train_content_list)
readUnlabelTrainData()
all_train_len = len(train_content_list)
readTestData()
content_list = train_content_list + test_content_list
print("Read Data Finished!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
# ------------------------------------------------------------------------------------
# 去停用词，转小写，词性还原
parseStopWord()
print("parseStopWord Finished!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
# ------------------------------------------------------------------------------------
# 训练word2vec模型
texts = [line.split() for line in seq]

num_features = 300    # Word vector dimensionality
min_word_count = 5   # Minimum word count
context = 10          # Context window size
model = word2vec.Word2Vec(texts, workers=10, size=num_features, min_count=min_word_count, window=context)

model_name = '{}features_{}minwords_{}context'.format(num_features, min_word_count, context)
model.wv.save_word2vec_format("./"+lower+model_name+".txt")
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# 计算每个句子的平均词向量
buildword2vec()
reviewFeatureVecs = np.zeros((len(seq), 300), dtype="float32")
counter = 0
for line in seq:
    list = line.split(" ")
    reviewFeatureVecs[counter]=makeFeatureVec(list,300)
    counter=counter+1
reduced_Feature = reviewFeatureVecs

#numpy数组截断不生成新的数组，只返回原数组的引用
train_final_seqs = reduced_Feature[0:label_train_len]
test_final_seqs = reduced_Feature[all_train_len:len(reduced_Feature)]
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# 保存处理好的结果，格式是python对象，可以直接用pickle读取
with open(lower+model_name+"_medium_sentences.pickle", 'wb') as f:
    pickle.dump(seq[0:label_train_len], f)  # 训练集句子
    pickle.dump(train_label_list, f) # 训练集标签
    pickle.dump(seq[all_train_len:len(seq)], f) # 测试集句子
    pickle.dump(test_id_list, f) # 测试集ID

with open(lower+model_name+"_medium_vectors.pickle", 'wb') as f:
    pickle.dump(train_final_seqs, f)  # 训练集句子向量
    pickle.dump(train_label_list, f) # 训练集标签
    pickle.dump(test_final_seqs, f) # 测试集句子向量
    pickle.dump(test_id_list, f) # 测试集ID
