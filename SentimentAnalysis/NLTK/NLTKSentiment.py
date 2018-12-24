# -*- coding: utf-8 -*- 
from nltk.classify import NaiveBayesClassifier
import pandas as pd
import csv

def preprocess(s):
    return {word: True for word in s.lower().split()}

#使用pandas读取训练集
dataSetPath = "../labeledTrainData.csv"
train = pd.read_csv(dataSetPath,encoding="ISO-8859-1")
#将读取出来的数据转为DataFram结构
df = pd.DataFrame(train)

#训练结合初始化
training_data = []
#计算数据的有多少行
lineCount = df.size//3
for i in range (0,lineCount):
    id = df.iloc[i][0]#id
    label = df.iloc[i][1]#label
    review = df.iloc[i][2]#review
    training_data.append([preprocess(review),label])
#print(training_data)
model = NaiveBayesClassifier.train(training_data)

print('train success')

#设置结果输出文件位置
resultPath =  "result.csv"
csv_file = open(resultPath,"w+")
csv_writer = csv.writer(csv_file)
#文件头
head = ("id","sentiment")
csv_writer.writerow(head)

#读取测试集合
testSetPath = "../testData.csv"
test = pd.read_csv(testSetPath,encoding="ISO-8859-1")
test_df = pd.DataFrame(test)

testCount = test_df.size//2
for i in range(0,testCount):
    id = test_df.iloc[i][0]
    review = test_df.iloc[i][1]
    label = model.classify(preprocess(review))
    result_list = (id,label)
    #print(result_list)
    csv_writer.writerow(result_list)

csv_file.close()
print(model.classify(preprocess('this is a bad movie')))
