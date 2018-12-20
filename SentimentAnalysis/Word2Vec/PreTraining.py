# -*- coding: utf-8 -*-
import os
import sys
import time
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models import word2vec
# reload(sys) 
# sys.setdefaultencoding('utf8')

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
def readUnlabelTrainData():
    #使用pandas读取测试集
    dataSetPath = "../unlabeledTrainData.csv"
    train = pd.read_csv(dataSetPath,encoding="ISO-8859-1")
    #将读取出来的数据转为DataFram结构
    df = pd.DataFrame(train)
    #计算数据的有多少行
    lineCount = df.size//2
    print(lineCount)
    for i in range (0,lineCount):
        try:
            if(len(df.iloc[i])==2):
                review = df.iloc[i][1]#review
                train_content_list.append(review)
        except:
            print(i)
        else:
            continue

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

print("Read Data Begin!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
readTrainData()
readTestData()
readUnlabelTrainData()
print("Read Data Finished!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))

content_list = train_content_list + test_content_list


parseStopWord()

f = open("Sentence.txt","w+")
for line in seq:
    f.write(line)
f.close()

f= open("Sentence.txt","r+")

f.close()
seq = word2vec.LineSentence("Sentence.txt")


num_features = 300    # Word vector dimensionality
min_word_count = 10   # Minimum word count
context = 10          # Context window size
print("model Training Begin!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
model = word2vec.Word2Vec(seq, workers=4, size=num_features, min_count=min_word_count, window=context)
print("model Training Finished!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))

model_name = '{}features_{}minwords_{}context'.format(num_features, min_word_count, context)
#print(model["like"])


# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model.wv.save_word2vec_format(model_name+".txt")


