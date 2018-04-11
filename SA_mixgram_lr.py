from SentimentAnalyze import generate_raw_data
from SentimentAnalyze import get_file_src_list
from SentimentAnalyze import partition_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import nltk
from sklearn.linear_model import LogisticRegression

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
    with open('./data/corpus.txt', 'w+', encoding='utf-8') as f:
        for x in corpus:
            f.write(x.x + '\n')
    pickle.dump(train,open('./data/train_t.pkl','wb+'),protocol=True)
    pickle.dump(val, open('./data/val_t.pkl', 'wb+'), protocol=True)
    pickle.dump(test_data, open('./data/test_t.pkl', 'wb+'), protocol=True)

def gen_fea_for_all_data():
    corpus=[]
    with open('./data/corpus.txt','r',encoding='utf-8') as f:
        for line in f:
            corpus.append(line)
    fea_mapper=TfidfVectorizer(ngram_range=(1,3),dtype=np.float32,tokenizer=nltk.tokenize.word_tokenize)
    fea_mapper.fit(corpus)
    train=pickle.load(open('./data/train_t.pkl', 'rb'))
    test = pickle.load(open('./data/test_t.pkl', 'rb'))
    val= pickle.load(open('./data/val_t.pkl', 'rb'))
    x_train=[t.x for t in train]
    y_train=[t.y for t in train]
    x_train=fea_mapper.transform(x_train)
    pickle.dump([x_train,y_train],open('./data/train_t_fea','wb+'),protocol=True)
    x_test = [t.x for t in test]
    y_test = [t.y for t in test]
    x_test = fea_mapper.transform(x_test)
    pickle.dump([x_test, y_test], open('./data/test_t_fea', 'wb+'), protocol=True)
    x_val = [t.x for t in val]
    y_val = [t.y for t in val]
    x_val = fea_mapper.transform(x_val)
    pickle.dump([x_val, y_val], open('./data/val_t_fea', 'wb+'), protocol=True)

def evaluate(y_pred, y_true):
    total_num = len(y_true)
    correct_num = 0.0
    for i in range(0, len(y_true)):
        if y_pred[i] == y_true[i]:
            correct_num += 1
    return correct_num / total_num

def train_model():
    x_train,y_train = pickle.load(open('./data/train_t_fea', 'rb'))
    x_val,y_val = pickle.load(open('./data/val_t_fea', 'rb'))
    clf = LogisticRegression(C=1e100, max_iter=100)
    clf.fit(x_train, y_train)
    result = clf.predict(x_val)
    print(evaluate(result, y_val))

def test():
    x_train,y_train = pickle.load(open('./data/train_t_fea', 'rb'))
    x_val,y_val = pickle.load(open('./data/val_t_fea', 'rb'))
    x_test,y_test = pickle.load(open('./data/test_t_fea', 'rb'))
    #x_train = np.concatenate([np.array(x_train),np.array(x_val)],axis=0)
    #y_train = np.concatenate([np.array(y_train) , np.array(y_val)],axis=0)
    clf = LogisticRegression(C=100000, max_iter=100)
    clf.fit(x_train, y_train)
    result = clf.predict(x_test)
    print(evaluate(result, y_test))

if __name__=='__main__':
    #preprocess1()
    #gen_fea_for_all_data()
    #train_model()
    test()
    print('all work has finished')
