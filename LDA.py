import jieba
from gensim import corpora, models
import os
import numpy as np


def get_stop_words_set(file_name):
    with open(file_name,'r',encoding='utf-8') as file:
        return set([line.strip() for line in file])

def get_words_list(datas,stop_word_file):
    stop_words_set = get_stop_words_set(stop_word_file)
    print("共计导入 %d 个停用词" % len(stop_words_set))
    word_list = []
    for line in datas:
        tmp_list = list(jieba.cut(line.strip(),cut_all=False))
        word_list.append([term for term in tmp_list if str(term) not in stop_words_set]) #注意这里term是unicode类型，如果不转成str，判断会为假
    return word_list

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

def load_data(src_list):
    datas=[]
    for src in src_list:
        with open(src,'r',encoding='utf-8') as f:
            all_lines=''
            for line in f:
                all_lines+=line.strip()
            datas.append(all_lines)
    return datas

if __name__ == '__main__':
    raw_msg_file_dir = get_file_src_list(parent_path=r"E:\Learning\课程\自然语言处理与智能搜索\PeoplePaper1946-1949")
    datas=load_data(raw_msg_file_dir)
    stop_word_file = "./data/stopword.txt"
    word_list = get_words_list(datas,stop_word_file) #列表，其中每个元素也是一个列表，即每行文字分词后形成的词语列表
    word_dict = corpora.Dictionary(word_list)  #生成文档的词典，每个词与一个整型索引值对应
    corpus_list = [word_dict.doc2bow(text) for text in word_list] #词频统计，转化成空间向量格式
    train_data=corpus_list[0:-3]
    lda = models.ldamodel.LdaModel(corpus=train_data,id2word=word_dict,num_topics=50,alpha='auto')
    output_file = './lda/lda_output.txt'
    test_data=corpus_list[-3:]
    with open(output_file,'w',encoding='utf-8') as f:
        for pattern in lda.show_topics(num_topics=50,num_words=100):
            f.write(str(pattern[0])+"    "+pattern[1]+'\n')
    train_topic=lda.inference(train_data)[0]
    test_topic=lda.inference(test_data)[0]
    for j in range(0,len(test_topic)):
        cosine_and_index=[]
        for i in range(0,len(train_data)):
            cosine_and_index.append([np.dot(test_topic[j],train_topic[i]),i])
        cosine_and_index.sort(reverse=True)
        print('id:'+str(j))
        print(datas[-3+j])
        print("similar documents:")
        print(datas[cosine_and_index[0][1]])
        print('')
        print(datas[cosine_and_index[1][1]])
        print('')

