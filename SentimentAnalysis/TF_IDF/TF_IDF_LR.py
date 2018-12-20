import nltk
from nltk.corpus import stopwords
import pandas as pd
import csv
import time
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV



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


print("Read Data begin!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
readTrainData()
readTestData()
print("Read Data Finished!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
content_list = train_content_list + test_content_list
parseStopWord()

tfidf = TFIDF(min_df=2, # 最小支持度为2
           max_features=None,
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1, 3),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english') # 去掉英文停用词

# 合并训练和测试集以便进行TFIDF向量化操作
print("TF_IDF Train Begin!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
tfidf.fit(seq)
data_all = tfidf.transform(seq)
# 恢复成训练集和测试集部分
train_x = data_all[:len(train_content_list)]
test_x = data_all[len(train_content_list):]

print("TF_IDF Train Finished!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))

# 设定grid search的参数
grid_values = {'C':[30]}
# 设定打分为roc_auc
LR_model = GridSearchCV(LR(penalty = 'l2', dual = True, random_state = 0), grid_values, scoring = 'roc_auc', cv = 20)
print("LR fit begin!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
LR_model = LR_model.fit(train_x, train_label_list)
print("LR fit end!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
print("Pretiction begin!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
predicted = LR_model.predict(test_x)
print("Pretiction end!"+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))

#设置结果输出文件位置
resultPath =  "TF_IDF_LRresult.csv"
csv_file = open(resultPath,"w+")
csv_writer = csv.writer(csv_file)
#文件头
head = ("id","sentiment")
csv_writer.writerow(head)

for i in range (0,len(test_content_list)):
    prediction = (test_id_list[i],predicted[i])
    csv_writer.writerow(prediction)

csv_file.close()



