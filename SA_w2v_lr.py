import sklearn as sk
from SentimentAnalyze import Data
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression


def get_sentence_vec(sentence, embedding, divide_sen_len=False):
    vec = np.array([0.0] * embedding.shape[1])
    for word in sentence:
        vec += embedding[word]
    if divide_sen_len:
        vec = vec / len(sentence)
    return vec


def evaluate(y_pred, y_true):
    total_num = len(y_true)
    correct_num = 0.0
    for i in range(0, len(y_true)):
        if y_pred[i] == y_true[i]:
            correct_num += 1
    return correct_num / total_num


def test():
    train = pickle.load(open('./data/train.pkl', 'rb'))
    val = pickle.load(open('./data/val.pkl', 'rb'))
    test = pickle.load(open('./data/test.pkl', 'rb'))
    embedding = pickle.load(open('./data/embedding.pkl', 'rb'))
    x_train = [get_sentence_vec(t.x, embedding) for t in train] + [get_sentence_vec(t.x, embedding) for t in val]
    y_train = [t.y for t in train] + [t.y for t in val]
    x_test = [get_sentence_vec(t.x, embedding) for t in test]
    y_test = [t.y for t in test]
    clf = LogisticRegression(C=10000, max_iter=100)
    clf.fit(x_train, y_train)
    result = clf.predict(x_test)
    print(evaluate(result, y_test))


def train_model():
    train = pickle.load(open('./data/train.pkl', 'rb'))
    val = pickle.load(open('./data/val.pkl', 'rb'))
    embedding = pickle.load(open('./data/embedding.pkl', 'rb'))
    x_train = [get_sentence_vec(t.x, embedding) for t in train]
    y_train = [t.y for t in train]
    x_val = [get_sentence_vec(t.x, embedding) for t in val]
    y_val = [t.y for t in val]
    clf = LogisticRegression(C=100000, max_iter=100)
    clf.fit(x_train, y_train)
    result = clf.predict(x_val)
    print(evaluate(result, y_val))


if __name__ == '__main__':
    #train_model()
    test()
    #def
        #Y = 1
        #fun_x = lambda x : x + Y
        #return fun_x
    #list_x = [1, 2, 3]
    #print(list(map(fun_x, list_x)))
