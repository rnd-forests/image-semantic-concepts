import numpy as np

word_list = np.load('embedding/word_list.npy')
word_list = word_list.tolist()
word_list = [word.decode('utf-8') for word in word_list]
word_vectors = np.load('embedding/word_vectors.npy')


def word_to_vec(word):
    try:
        idx = word_list.index(word)
    except ValueError:
        idx = 399999
    return word_vectors[idx]


print(word_to_vec('awesome'))
