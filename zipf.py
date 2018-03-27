import thulac
import matplotlib.pyplot as plt
import os
import math
import csv


class Tokenizer:
    def __init__(self):
        self.user_dict = None
        self.model_path = None  # 默认为model_path
        self.T2S = True  # 繁简体转换
        self.seg_only = True  # 只进行分词
        self.filt = False  # 去停用词
        self.tokenizer = thulac.thulac(user_dict=self.user_dict, model_path=self.model_path, T2S=self.T2S,
                                       seg_only=self.seg_only, filt=self.filt)

    def parser(self, text):
        return self.tokenizer.cut(text, text=True)  # 返回文本


class Word_Frequent_Dict:
    def __init__(self):
        self.word_occurrences_dict = {}
        self.total_occurrences = 0.0

    def get_word_frequent(self,word):
        if word in self.word_occurrences_dict:
            return self.word_occurrences_dict[word] / self.total_occurrences
        else:
            return 0.0

    def count_one_word(self,word):
        if word in self.word_occurrences_dict:
            self.word_occurrences_dict[word] += 1.0
        else:
            self.word_occurrences_dict[word] = 1.0
        self.total_occurrences += 1.0

    def get_word_occurrence(self,word):
        if word in self.word_occurrences_dict:
            return self.word_occurrences_dict[word]
        else:
            return 0.0

    def clear(self):
        self.total_occurrences = 0.0
        self.word_occurrences_dict.clear()

    def get_vocab_size(self):
        return len(self.word_occurrences_dict)

    def get_dict_keys(self):
        return self.word_occurrences_dict.keys()

    def get_frequent_list_desc(self):
        fre_list = []
        for key in self.word_occurrences_dict.keys():
            fre_list.append(self.word_occurrences_dict[key] / self.total_occurrences)
        fre_list.sort(reverse=True)
        return fre_list

def save(list,src='model/list.csv'):
    if not os.path.isdir('model'):
        os.mkdir('model')
    with open(src,mode='w+',encoding='utf-8') as file:
        writer=csv.writer(file)
        for l in list:
            writer.writerow(l)


def get_file_src_list(parent_path, file_type='.txt'):
    files = os.listdir(parent_path)
    src_list = []
    for file in files:
        absolute_path = os.path.join(parent_path, file)
        if os.path.isdir(absolute_path):
            src_list+=get_file_src_list(absolute_path)
        elif file.endswith(file_type):
            src_list.append(absolute_path)
    return src_list


def stat_corpus(src, word_level=True, char_level=True, word_dict=None, char_dict=None,tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer()
    if word_dict is None:
        word_dict = Word_Frequent_Dict()
    if char_dict is None:
        char_dict = Word_Frequent_Dict()
    with open(src, 'r', encoding='utf-8') as file:
        for line in file:
            if word_level:
                word_list = tokenizer.parser(line.strip('\r\n').strip('\n')).split(' ')
                for word in word_list:
                    word_dict.count_one_word(word)
            if char_level:
                char_list = list(line.strip('\r\n').strip('\n'))
                for char in char_list:
                    char_dict.count_one_word(char)
    return word_dict, char_dict


def plot(fre_list, title='default', xlabel='x', ylabel='y'):
    x = [math.log2(i) for i in range(1, len(fre_list)+1)]
    y = [math.log2(i) for i in fre_list]
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def cal_zipf_curve():
    src_list = get_file_src_list(parent_path=r"E:\Learning\课程\自然语言处理与智能搜索\PeoplePaper1946-1949")
    word_dict = Word_Frequent_Dict()
    char_dict = Word_Frequent_Dict()
    tokenizer=Tokenizer()
    for src in src_list:
        word_dict, char_dict = stat_corpus(src, word_dict=word_dict, char_dict=char_dict,tokenizer=tokenizer)
    word_frequent_list = word_dict.get_frequent_list_desc()
    char_frequent_list = char_dict.get_frequent_list_desc()
    save([word_frequent_list,char_frequent_list])
    plot(word_frequent_list, title='word level', xlabel='x=log2(rank)', ylabel='y=log2(frequency)')
    plot(char_frequent_list, title='character level', xlabel='x=log2(rank)', ylabel='y=log2(frequency)')

if __name__=='__main__':
    cal_zipf_curve()