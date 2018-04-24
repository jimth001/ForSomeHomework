import math
import os
import numpy as np

class BiGram_LM:
    def __init__(self):
        self.start_dict = {}
        self.bigram_dict = {}
        self.total_start_num = 0.0

    def fit(self, sentences):
        for sen in sentences:
            first = True
            last_char = None
            for character in sen:
                if first:
                    if character in self.start_dict:
                        self.start_dict[character] += 1
                    else:
                        self.start_dict[character] = 1.0
                    first = False
                    self.total_start_num += 1
                else:
                    if last_char in self.bigram_dict:
                        adict = self.bigram_dict[last_char][1]
                        if character in adict:
                            adict[character] += 1
                        else:
                            adict[character] = 1.0
                        self.bigram_dict[last_char][0] += 1
                    else:
                        adict = {}
                        adict[character] = 1.0
                        self.bigram_dict[last_char] = [1.0, adict]
                last_char = character

    def get_log_prop(self, sentence):
        rst = 0.0
        first = True
        last_character = None
        for character in sentence:
            if first:
                if character in self.start_dict:
                    rst += math.log2(self.start_dict[character] / self.total_start_num)
                else:
                    rst += math.log2(1 / (self.total_start_num + 1))
            else:
                if last_character in self.bigram_dict:
                    tnum, adict = self.bigram_dict[last_character]
                    if character in adict:
                        rst += math.log2(adict[character] / tnum)
                    else:
                        rst += math.log2(1 / (tnum + 1))
                else:
                    rst += math.log2(1 / (len(self.bigram_dict) + 1))
            last_character = character
        return -rst/len(sentence)


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


def get_sentences(src_list):
    sentences=[]
    for src in src_list:
        with open(src,'r',encoding='utf-8') as f:
            for line in f:
                sentences+=line.split()
    return sentences

if __name__ == '__main__':
    src_list=get_file_src_list(parent_path=r"E:\Learning\课程\自然语言处理与智能搜索\PeoplePaper1946-1949")
    sentences=get_sentences(src_list)
    bigram_lm=BiGram_LM()
    bigram_lm.fit(sentences[0:-10])
    entropy=[]
    for s in sentences[-10:]:
        entropy.append(bigram_lm.get_log_prop(s))
    print(entropy)
    print(np.array(entropy).min())
    
