from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from numpy.linalg import svd, pinv
import numpy as np


def lsa(s, v, sentences):
    b = s.shape
    c = v.shape
    if c[0] > b[0]:
        new_v = v[0:b[0]]
    else:
        new_v = v
    new_s = np.mat(np.diag(s))
    pinv_v = pinv(new_v)
    convert_mat = np.matmul(pinv_v, new_s)
    return np.matmul(sentences, convert_mat)


def cos_sim(vector1, vector2, append_data=None):
    return [float(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))), append_data]


if __name__ == "__main__":
    context = ['Romeo and Juliet',
               'Juliet:o happy dagger!',
               'Romeo died by dagger',
               'Live free a die,that is the New-Hampshine\'s motto',
               'Did you know,New-Hampshine is in New England']
    query = ['died dagger']
    tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk.tokenize.word_tokenize)
    tfidf_vectorizer.fit(context)
    tfidf_query = tfidf_vectorizer.transform(query)
    tfidf_context = tfidf_vectorizer.transform(context)
    tfidf_query = tfidf_query.todense()
    tfidf_context = tfidf_context.todense()
    u, s, v = svd(tfidf_context)
    lsa_context = lsa(s, v, tfidf_context)
    lsa_query = lsa(s, v, tfidf_query)
    sim = []
    for i in range(0, len(lsa_context)):
        sim.append(cos_sim(lsa_query[0], lsa_context[i].T, context[i]))
    sim.sort(key=lambda x: x[0], reverse=True)
    for t in sim:
        print(t)
