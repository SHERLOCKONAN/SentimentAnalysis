import numpy as np
import tensorflow as tf
import pandas as pd
from string import punctuation
from collections import Counter

reviews = []
labels = []

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
    reviews.append(review)
    labels.append(label)

print(reviews[:2])
print(labels[:2])
    
all_reviews = '\n'.join(reviews)
all_text = ''.join([c for c in all_reviews if c not in punctuation])
reviews = all_text.split('\n')
words = all_text.split()

print(reviews[:2])
print(len(words))

counts = Counter(words)
vocab = sorted(counts,key=counts.get,reverse = True)
vocab_to_int = {word:ii for ii,word in enumerate(vocab,1)}

reviews_ints = []

for review in reviews:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])

print(len(reviews_ints))
print(reviews_ints[1])

review_lens = Counter([len(x) for x in reviews_ints])

print('Zero-length reviews: {}'.format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

revice_len_zero = 0

for i,review in enumerate(reviews_ints,0):
    if len(review)==0:
        revice_len_zero = i
print(revice_len_zero)

reviews_ints = [review_int for review_int in reviews_ints if len(review_int)>0]



seq_len = 200
#处理多余200个单词的评论
reviews_ints = [review[:200] for review in reviews_ints]
#处理少于200个单词的评论
features = []
for review in reviews_ints:
    if len(review) < seq_len : 
        s = []
        for i in range(seq_len - len(review)):
            s.append(0)
        s.extend(review)
        features.append(s)
    else:
        features.append(review)
features = np.array(features)


split_frac = 0.8

from sklearn.model_selection import train_test_split
train_x, val_x = train_test_split(features, test_size = 1 - split_frac, random_state = 0)
train_y, val_y = train_test_split(labels, test_size = 1 - split_frac, random_state = 0)
val_x, test_x = train_test_split(val_x, test_size = 0.5, random_state = 0)
val_y, test_y = train_test_split(val_y, test_size = 0.5, random_state = 0)


print("\t\tFeatures Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
    "\nValidation set: \t{}".format(val_x.shape),
    "\nTest set: \t\t{}".format(test_x.shape))

lstm_size = 256
lstm_layers = 1
batch_size = 500
learning_rate = 0.1

n_words = len(vocab_to_int) + 1
#加1是因为字典从1开始，我们用0来填充 
# #创建图对象
graph = tf.Graph() 
#像图中添加节点
with graph.as_default():    
    inputs = tf.placeholder(tf.int32, [None, None], name = 'inputs')    
    labels = tf.placeholder(tf.int32, [None, None], name = 'labels')   
    keep_prod = tf.placeholder(tf.float32, name = 'keep_prod')


#嵌入向量的大小(嵌入层单元个数)
embed_size = 300

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)


with graph.as_default():
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prod)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

with graph.as_default():    
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)    
    cost = tf.losses.mean_squared_error(labels, predictions)        
    
    optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

with graph.as_default():    
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)    
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_batchs(x, y, batch_size = 100):
    n_batchs = len(x) // batch_size
    x, y = x[:n_batchs * batch_size], y[:n_batchs * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


epochs = 10

with graph.as_default():
    saver = tf.train.Saver()


with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        
        for ii, (x, y) in enumerate(get_batchs(train_x, train_y, batch_size), 1):
            feed = {inputs : x,
                   labels : (np.array(y))[:,None],
                   keep_prod : 0.5,
                   initial_state : state}

            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict = feed)
        
            if iteration % 5 == 0:
                print("Epoch: {}/{}".format(e, epochs),
                    "Iteration: {}".format(iteration),
                    "Train loss: {:.3f}".format(loss))

            if iteration % 25 == 0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batchs(val_x, val_y, batch_size):
                    feed = {
                        inputs : x,
                        labels : (np.array(y))[:,None],
                        keep_prod : 1,
                        initial_state : val_state
                    }
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict = feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            
            iteration += 1

    saver.save(sess, 'checkpoints/sentiment.ckpt')


test_acc = []
with tf.Session(graph = graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batchs(test_x, test_y, batch_size), 1):
        feed = {inputs : x,
                #    labels : y[:, None],
                labels : (np.array(y))[:,None],
                keep_prod : 1,
                initial_state : test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict = feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))



