import os
import string

import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize

import config
from captions import Captions
from helpers import save_variables, load_variables


class Vocabulary:
    def __init__(self, caps_file, size=None, save_file=None):
        self.size = size
        self.vocab = None
        self.caps_file = caps_file
        self.save_file = save_file

    def build(self):
        pos = list()
        caption_processor = Captions(self.caps_file)
        image_ids = caption_processor.get_img_ids()

        for i in image_ids:
            print(i)
            caps = caption_processor.get_captions(i)
            tmp = [pos_tag(word_tokenize(str(a).lower())) for a in caps]
            pos.append(tmp)

        pos_mapping = self.__pos_mapping()
        pos = [p3 for p1 in pos for p2 in p1 for p3 in p2]
        pos = [(w, pos_mapping.get(t)) for (w, t) in pos if pos_mapping.get(t) is not None]
        _vocab = Counter(elem for elem in pos)
        _vocab = _vocab.most_common()

        word = [w for ((w, t), c) in _vocab]
        word = [w for w in word if w not in self.__stop_words()]
        pos = [t for ((w, t), c) in _vocab]
        count = [c for ((_, _), c) in _vocab]

        poss = []
        counts = []
        words = sorted(set(word))
        for j in range(len(words)):
            indexes = [i for i, x in enumerate(word) if x == words[j]]
            pos_tmp = [pos[i] for i in indexes]
            count_tmp = [count[i] for i in indexes]
            idx = np.argmax(count_tmp)
            poss.append(pos_tmp[idx])
            counts.append(sum(count_tmp))

        idx = np.argsort(counts)
        idx = idx[::-1]
        words = [words[i] for i in idx]
        poss = [poss[i] for i in idx]
        counts = [counts[i] for i in idx]

        no_punctuation = [i for (i, x) in enumerate(words) if x not in self.__punctuations()]
        words = [words[i] for i in no_punctuation]
        counts = [counts[i] for i in no_punctuation]
        poss = [poss[i] for i in no_punctuation]

        self.vocab = {'words': words, 'counts': counts, 'poss': poss}

    def store(self):
        if os.path.exists(self.save_file) or self.size is None:
            return

        if self.save_file is None:
            raise ValueError('Must specify the path for storing vocabulary.')

        self.build()
        _vocab = self.__get_top_k(self.size)
        save_variables(self.save_file, _vocab)

    def get(self):
        if self.vocab is None:
            self.build()
        return self.vocab

    def load(self):
        if not os.path.exists(self.save_file):
            raise ValueError('Vocabulary file does not exist.')
        return load_variables(self.save_file)

    def __get_top_k(self, k):
        v = dict()
        for key in self.vocab.keys():
            v[key] = self.vocab[key][:k]
        return v

    @staticmethod
    def __stop_words():
        return stopwords.words('english')

    @staticmethod
    def __punctuations():
        return list(string.punctuation) + ["--", "-LRB-", "-RRB-", "-LCB-", "-RCB-", "..."]

    @staticmethod
    def __pos_mapping():
        return {'NNS': 'NN', 'NNP': 'NN', 'NNPS': 'NN', 'NN': 'NN',
                'VB': 'VB', 'VBD': 'VB', 'VBN': 'VB', 'VBZ': 'VB', 'VBP': 'VB', 'VBG': 'VB',
                'JJR': 'JJ', 'JJS': 'JJ', 'JJ': 'JJ'}


if __name__ == "__main__":
    vocab = Vocabulary(config.CAPTIONS_FILE, config.VOCAB_SIZE, config.VOCABULARY_FILE)
    vocab.store()
