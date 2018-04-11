import sklearn as sk
import os
import word2vec
import jieba
import numpy as np
from scipy import sparse
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer
import time
import pickle
from datetime import timedelta

from sklearn.linear_model import LogisticRegression
data_path='./data/'
class NNModel:
    def __init__(self,embedding):
        self.learning_rate=5e-3
        self.batch_size=32
        self.epoch_num=6
        self.dropout_keep_prob=0.5
        self.embedding=embedding
        self.vocab_size=embedding.shape[0]
        self.input_x = tf.placeholder(tf.int64, [None, None], name='input_x')  # placeholder只存储一个batch的数据
        self.x_sequence_len = tf.placeholder(tf.int64, [None], name='x_sequence_len')
        self.embedding_ph=tf.placeholder(tf.float32,[self.embedding.shape[0],self.embedding.shape[1]],name='embedding')
        self.input_y = tf.placeholder(tf.int64, [None], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.inputs_data=[self.input_x,self.x_sequence_len]
        #
        self.tensorboard_dir='./tensorboard'
        self.save_dir='./model/'
        self.print_per_batch=32
        self.save_per_batch=64
        self.require_improvement=6400

    def __get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def build_basic_rnn_model(self,rnn_unit_num=32,dense_layer_unit_num=8,class_num=2,reg_para=0.0):
        with tf.device('/cpu:0'):  # 指定cpu0执行
            word_embedding = tf.get_variable(name='embedding', shape=self.embedding.shape,dtype=tf.float32,trainable=True)
            self.embedding_init = word_embedding.assign(self.embedding_ph)
            x_embedding=tf.nn.embedding_lookup(word_embedding,self.input_x)
        with tf.name_scope("rnn"):
            gru_cell = rnn.GRUCell(rnn_unit_num, kernel_initializer=tf.orthogonal_initializer())
            cell_with_dropout=rnn.DropoutWrapper(gru_cell, output_keep_prob=self.keep_prob)
            _,state=tf.nn.dynamic_rnn(cell=cell_with_dropout,inputs=x_embedding,sequence_length=self.x_sequence_len,dtype=tf.float32)
        with tf.name_scope("dense_layers"):
            fc = tf.layers.dense(state, dense_layer_unit_num, name='fc1', activation=tf.nn.tanh,
                                kernel_initializer=xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_para),
                                bias_initializer=tf.zeros_initializer(),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(reg_para))
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            self.logits = tf.layers.dense(fc, class_num, name='fc2', activation=tf.nn.tanh,
                                        kernel_initializer=xavier_initializer(),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_para),
                                        bias_initializer=tf.zeros_initializer(),
                                        bias_regularizer=tf.contrib.layers.l2_regularizer(reg_para))
            self.y_pred_value = tf.nn.softmax(self.logits)  # 输出概率值(相似度值)
            self.y_pred_class = tf.argmax(self.y_pred_value, 1)  # 预测类别
        with tf.name_scope("optimize"):
            # 损失函数：交叉熵
            cross_entropy =tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits+1e-10,labels=self.input_y)#这个可以不用把y变成one-hot的向量
            self.loss = tf.reduce_mean(cross_entropy) # 对tensor所有元素求平均
            # 优化器。指定优化方法，学习率，最大化还是最小化，优化目标函数为交叉熵
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("evaluate_metrics"):
            # 准确率
            correct_pred = tf.equal(self.input_y, self.y_pred_class)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def _evaluate_without_predict_result(self, input_data, target):
        batches = self.batch_iter([input_data], target, self.batch_size,shuffle=False)
        total_loss = 0.0
        total_acc = 0.0
        total_len = len(target)
        for batch_data, batch_target in batches:
            batch_len = len(batch_target)
            feed_dict = self.feed_data(inputs_data=batch_data, keep_prob=1.0, target=batch_target)
            loss, acc= self.session.run([self.loss, self.acc],feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len
        return total_loss / total_len, total_acc / total_len

    def feed_data(self, inputs_data, keep_prob, target=None):
        # 每个具体的model要注意定义placeholder的顺序和batch_iter返回的数据的顺序要一致，对应。
        feed_dict = {}
        for i in range(len(self.inputs_data)):
            feed_dict[self.inputs_data[i]] = inputs_data[i]
        feed_dict[self.keep_prob] = keep_prob
        if not target is None:
            feed_dict[self.input_y] = target
        return feed_dict

    def evaluate(self,input_data,target,model_path):
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4  # 只分配50%的显存
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.session=sess
            saver.restore(sess, model_path)
            print(self._evaluate_without_predict_result(input_data,target))

    def train_model(self,train_x,train_label,val_x,val_label,continue_train=False,debug=False,previous_model_path=None):
        start_time = time.time()
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        # 配置 Saver
        saver = tf.train.Saver()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        ##############################################################
        print(str(self.__get_time_dif(start_time)) + "trainning and evaluating...")
        total_batch = 0
        best_acc_val = 0.0
        last_improved = 0
        flag = False  # 停止标志
        # 创建session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4  # 只分配50%的显存
        config.gpu_options.allow_growth=True
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4,allow_growth=True)
        with tf.Session(config=config) as sess:
            self.session=sess
            if debug:
                # summary信息包括loss和accuracy，并进行merge。以后run一个batch指定fetch的东西为merged_summary就可以得到loss和accuracy
                tf.summary.scalar("loss", self.loss)
                tf.summary.scalar("accuracy", self.acc)
                merged_summary = tf.summary.merge_all()
                writer = tf.summary.FileWriter(self.tensorboard_dir, graph=self.session.graph)
            if continue_train is False:
                sess.run(tf.global_variables_initializer())
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: self.embedding})
            else:
                saver.restore(sess, previous_model_path)
            self.session.graph.finalize()
            for epoch in range(self.epoch_num):
                print("epoch:" + str(epoch + 1))
                batch_train = self.batch_iter([train_x], train_label, batch_size=self.batch_size, shuffle=True)
                for batch_data, batch_target in batch_train:
                    feed_dict = self.feed_data(inputs_data=batch_data, target=batch_target,
                                               keep_prob=self.dropout_keep_prob)
                    if debug:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        s = self.session.run([self.optim, merged_summary],
                                             feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                    else:
                        s = self.session.run([self.optim], feed_dict=feed_dict)
                    if total_batch == 0:  # 初次存一下。因为测试代码时数据很少，可能到不了要保存的batch数就结束了。
                        saver.save(sess=self.session, save_path=self.save_dir + "model.ckpt")  # 保存当前的训练结果
                    if total_batch % self.save_per_batch == 0 and debug:  # 每多少轮次将训练结果写入tensorboard scalar
                        writer.add_run_metadata(run_metadata, 'step{}'.format(total_batch))
                        writer.add_summary(s[1], total_batch)  # 传入summary信息和当前的batch数
                    if total_batch > 0 and total_batch % self.print_per_batch == 0:  # 每多少轮次输出在训练集和验证集上的性能
                        feed_dict[self.keep_prob] = 1.0  # 在验证集上验证时dropout的概率改为0
                        # 算一下在这个train_batch上的loss和acc
                        loss_train, acc_train = self.session.run([self.loss, self.acc],
                                                                 feed_dict=feed_dict)
                        loss_val, acc_val = self._evaluate_without_predict_result(val_x, val_label)  # 验证，得到loss和acc
                        if acc_val > best_acc_val:
                            # 保存最好结果
                            best_acc_val = acc_val
                            last_improved = total_batch
                            saver.save(sess=self.session,
                                       save_path=self.save_dir + str(total_batch) + 'model.ckpt')  # 保存当前的训练结果
                            improved_str = '*'
                        else:
                            improved_str = ''
                        time_dif = self.__get_time_dif(start_time)
                        msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                              + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                        print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
                    total_batch += 1
                    if total_batch - last_improved > self.require_improvement:
                        # 早停：验证集正确率长期不提升，提前结束训练
                        print("No optimization for a long time, auto-stopping...")
                        flag = True
                        break  # 跳出batch
                if flag:  # 跳出epoch
                    break
        self.session=None

    def batch_iter(self, input_data, target=None, batch_size=64,padding=0,shuffle=False):
        #该模型的数据结构为[[q_list],[r_list]]

        assert not input_data is None,"input_data is None"
        data_len = len(input_data[0])
        num_batch = int((data_len - 1) / batch_size) + 1
        if shuffle:#target must be not none
            indices = np.random.permutation(np.arange(data_len))
        else:
            indices = range(data_len)
        x_shuffle = [input_data[0][i] for i in indices]
        x_seq_len = [len(x_shuffle[i]) for i in range(len(x_shuffle))]
        if target is None:
            for i in range(num_batch):
                start_id = i * batch_size
                end_id = min((i + 1) * batch_size, data_len)
                batch_x = x_shuffle[start_id:end_id]
                batch_x_seq_len = x_seq_len[start_id:end_id]
                x_max_len = max(batch_x_seq_len)
                for list in batch_x:
                    if len(list) < x_max_len:
                        list += [padding] * (x_max_len - len(list))
                yield [batch_x,batch_x_seq_len]
        else:
            y_shuffle = [target[i] for i in indices]
            for i in range(num_batch):
                start_id = i * batch_size
                end_id = min((i + 1) * batch_size, data_len)
                batch_y = y_shuffle[start_id:end_id]
                batch_x = x_shuffle[start_id:end_id]
                batch_x_seq_len = x_seq_len[start_id:end_id]
                x_max_len = max(batch_x_seq_len)
                for list in batch_x:
                    if len(list) < x_max_len:
                        list += [padding] * (x_max_len - len(list))
                yield [batch_x,batch_x_seq_len],batch_y


def get_file_src_list(parent_path, file_type='.txt'):
    files = os.listdir(parent_path)
    src_list = []
    for file in files:
        absolute_path = os.path.join(parent_path, file)
        if os.path.isdir(absolute_path):
            src_list += get_file_src_list(absolute_path)
        elif file.endswith(file_type):
            src_list.append(absolute_path)
    return src_list


class Data:
    def __init__(self,x,y):
        self.x=x#str or list
        self.y=y#int label 0 or 1
    def split(self):
        self.x=self.x.split(' ')
    def str2index(self,word_dict,with_unk=True):
        index=[]
        if with_unk:
            for s in self.x:
                if s in word_dict:
                    index.append(word_dict[s])
                else:
                    index.append(len(word_dict))
        else:
            for s in self.x:
                if s in word_dict:
                    index.append(word_dict[s])
        self.x=index

def generate_raw_data(src_list):
    data=[]
    for src in src_list:
        id, rating = os.path.basename(src).strip('.txt').split('_')
        with open(src,'r',encoding='utf-8') as f:
            text=''
            for line in f:
                text+=line
            if int(rating) > 5:
                data.append(Data(text,1))
            else:
                data.append(Data(text,0))
    return data

def pre_train_word_embedding():
    word2vec.word2vec('./data/corpus_for_w2v.txt', './data/word_embedding.bin', size=200, window=8, sample='1e-5',
                      cbow=0, save_vocab='./data/worddict', min_count=3)

def partition_dataset(data, n_fold=4, with_test=False):  # base on Conversation
    # 根据x，y划分n-fold交叉验证的数据集。如果with_test=True,则会划分测试集
    # 若输入x有n个m维特征。则x.shape is [n,data_num,m]
    assert n_fold >= 3, "n_flod must be bigger than 3"
    val_len = int(len(data) / n_fold)
    indices = np.random.permutation(np.arange(len(data)))
    data_shuffle = [data[i] for i in indices]
    val = data_shuffle[0:val_len]
    if with_test:
        test = data_shuffle[val_len:2 * val_len]
        train = data_shuffle[2 * val_len:]
        return train, val, test
    else:
        val = data_shuffle[0:val_len]
        train = data_shuffle[val_len:]
        return train, val

def load_word_embedding():
    # word_embedding:[clusters=None,vectors,vocab,vocab_hash]
    word_embedding = word2vec.load('./data/word_embedding.bin')
    return word_embedding

def preprocess1():
    neg_test = generate_raw_data(get_file_src_list(r'E:\Learning\课程\自然语言处理与智能搜索\aclImdb\test\neg'))
    pos_test = generate_raw_data(get_file_src_list(r'E:\Learning\课程\自然语言处理与智能搜索\aclImdb\test\pos'))
    neg_train = generate_raw_data(get_file_src_list(r'E:\Learning\课程\自然语言处理与智能搜索\aclImdb\train\neg'))
    pos_train = generate_raw_data(get_file_src_list(r'E:\Learning\课程\自然语言处理与智能搜索\aclImdb\train\pos'))
    unsup_data = generate_raw_data(get_file_src_list(r'E:\Learning\课程\自然语言处理与智能搜索\aclImdb\train\unsup'))
    train_data = neg_train + pos_train
    test_data = neg_test + pos_test
    train, val = partition_dataset(train_data, with_test=False)
    corpus = train_data + unsup_data
    with open('./data/corpus_for_w2v.txt', 'w+', encoding='utf-8') as f:
        for x in corpus:
            f.write(x.x + '\n')
    pre_train_word_embedding('./data/corpus_for_w2v.txt')


def preprocess_for_nn_model(with_unk=True):
    neg_test = generate_raw_data(get_file_src_list(r'E:\Learning\课程\自然语言处理与智能搜索\aclImdb\test\neg'))
    pos_test = generate_raw_data(get_file_src_list(r'E:\Learning\课程\自然语言处理与智能搜索\aclImdb\test\pos'))
    neg_train = generate_raw_data(get_file_src_list(r'E:\Learning\课程\自然语言处理与智能搜索\aclImdb\train\neg'))
    pos_train = generate_raw_data(get_file_src_list(r'E:\Learning\课程\自然语言处理与智能搜索\aclImdb\train\pos'))
    train_data = neg_train + pos_train
    test_data = neg_test + pos_test
    train, val = partition_dataset(train_data, with_test=False)
    embedding=load_word_embedding()
    for d in train:
        d.split()
        d.str2index(embedding.vocab_hash,with_unk)
    for d in val:
        d.split()
        d.str2index(embedding.vocab_hash,with_unk)
    for d in test_data:
        d.split()
        d.str2index(embedding.vocab_hash,with_unk)
    if with_unk is True:
        embedding.vectors = np.concatenate([embedding.vectors,np.array([[0.0]* embedding.vectors.shape[1]])],axis=0)
    pickle.dump(embedding.vectors, open('./data/embedding.pkl', 'wb+'), protocol=True)
    pickle.dump(train, open('./data/train.pkl', 'wb+'), protocol=True)
    pickle.dump(test_data, open('./data/test.pkl', 'wb+'), protocol=True)
    pickle.dump(val, open('./data/val.pkl', 'wb+'), protocol=True)


def load_labeled_bow(path):
    label=[]
    index_value=[]
    max=0
    with open(path,'r',encoding='ascii') as f:
        for line in f:
            one_index_value = []
            values=line.split(' ')
            label.append(int(values[0]))
            for t in values[1:]:
                tmp=t.split(':')
                one_index_value.append([int(tmp[0]),int(tmp[1])])
                if max<int(tmp[0]):
                    max=int(tmp[0])
            index_value.append(one_index_value)
    data=[]
    ceshi=sparse.dok_matrix(index_value)
    for one_index_value in index_value:
        vec=[0]*(max+1)
        for t in one_index_value:
            vec[t[0]]=t[1]
        data.append(vec)
    return data,label

def logistic_regression():
    train_src=r'E:\Learning\课程\自然语言处理与智能搜索\aclImdb\train\labeledBow.feat'
    test_src=r'E:\Learning\课程\自然语言处理与智能搜索\aclImdb\test\labeledBow.feat'
    train_data,train_labels=load_labeled_bow(train_src)
    test_data,test_labels=load_labeled_bow(test_src)
    clf=LogisticRegression(n_jobs=6)
    clf.fit(train_data,train_labels)
    result=clf.predict(test_data)
    #todo



def evaluate(pred,label,label_num=2):
    total_num=len(label)
    stat_per_label=[0]*label_num
    for i in range(0, len(label)):
        #todo
        pass


class TFIDF:
    def __init__(self):
        #todo
        pass


def use_nn_model():
    train = pickle.load(open('./data/train.pkl', 'rb'))
    val = pickle.load(open('./data/val.pkl', 'rb'))
    embedding=pickle.load(open('./data/embedding.pkl','rb'))
    #test = pickle.load(open('./data/test.seg.pkl', 'rb'))
    nn=NNModel(embedding)
    nn.build_basic_rnn_model()
    nn.train_model([t.x for t in train],[t.y for t in train],[t.x for t in val],[t.y for t in val])

def test():
    test = pickle.load(open('./data/test.pkl', 'rb'))
    embedding = pickle.load(open('./data/embedding.pkl', 'rb'))
    nn = NNModel(embedding)
    nn.build_basic_rnn_model()
    nn.evaluate([t.x for t in test],[t.y for t in test],model_path='')

if __name__=='__main__':
    use_nn_model()
    #test()
    print('all work has finished')