import re

import numpy as np
from collections import Counter
from nltk import pos_tag, word_tokenize

import config
from helpers import save_variables
from ordered_dict import DefaultListOrderedDict


class Captions:
    def __init__(self, file):
        self.file = file
        self.caps = DefaultListOrderedDict()
        self.load()

    def load(self):
        with open(self.file) as file:
            line = file.readline()
            while line:
                img, cap = re.split(r'#\d\t?', line, 1)
                img_id = img.split('.')[0]
                # if img_id not in self.caps:
                self.caps[img_id].append(cap)
                line = file.readline()

    def get_img_ids(self):
        ids = list(self.caps.keys())
        return ids

    def get_captions(self, img_id):
        return self.caps[img_id]

    def get_all_captions(self):
        return self.caps


def get_vocab_top_k(vocab, k):
    v = dict()
    for key in vocab.keys():
        v[key] = vocab[key][:k]
    return v


def get_vocab(caps, punctuations, mapping):
    image_ids = caps.get_img_ids()
    t = []

    for i in image_ids:
        print(i)
        anns = caps.get_captions(i)
        tmp = [pos_tag(word_tokenize(str(a).lower())) for a in anns]
        t.append(tmp)

    t = [t3 for t1 in t for t2 in t1 for t3 in t2]
    t = [(l, 'other') if mapping.get(r) is None else (l, mapping[r]) for (l, r) in t]
    vocab = Counter(elem for elem in t)
    vocab = vocab.most_common()

    word = [l for ((l, r), c) in vocab]
    pos = [r for ((l, r), c) in vocab]
    count = [c for ((l, r), c) in vocab]

    poss = []
    counts = []
    words = sorted(set(word))
    for j in range(len(words)):
        indexes = [i for i, x in enumerate(word) if x == words[j]]
        pos_tmp = [pos[i] for i in indexes]
        count_tmp = [count[i] for i in indexes]
        ind = np.argmax(count_tmp)
        poss.append(pos_tmp[ind])
        counts.append(sum(count_tmp))

    ind = np.argsort(counts)
    ind = ind[::-1]
    words = [words[i] for i in ind]
    poss = [poss[i] for i in ind]
    counts = [counts[i] for i in ind]

    non_punct = [i for (i, x) in enumerate(words) if x not in punctuations]
    words = [words[i] for i in non_punct]
    counts = [counts[i] for i in non_punct]
    poss = [poss[i] for i in non_punct]

    vocab = {'words': words, 'counts': counts, 'poss': poss}
    return vocab


def build_dictionary():
    mapping = {'NNS': 'NN', 'NNP': 'NN', 'NNPS': 'NN', 'NN': 'NN',
           'VB': 'VB', 'VBD': 'VB', 'VBN': 'VB', 'VBZ': 'VB', 'VBP': 'VB', 'VBP': 'VB', 'VBG': 'VB',
           'JJR': 'JJ', 'JJS': 'JJ', 'JJ': 'JJ', 'DT': 'DT', 'PRP': 'PRP', 'PRP$': 'PRP', 'IN': 'IN'}

    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    captions = Captions(config.CAPTIONS_PATH)
    vocab = get_vocab(captions, punctuations, mapping)
    vocab = get_vocab_top_k(vocab, config.VOCAB_SIZE)
    save_variables(config.DICTIONARY_PATH, vocab)
