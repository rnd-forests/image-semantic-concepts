import os

import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize

import config
from captions import Captions
from helpers import save_variables, load_variables


def get_top_k_vocab(vocab, k):
    v = dict()
    for key in vocab.keys():
        v[key] = vocab[key][:k]
    return v


def get_vocab(punctuations, mapping):
    pos = list()
    stop_words = stopwords.words('english')
    cap_loader = Captions(config.CAPTIONS_PATH)
    image_ids = cap_loader.get_img_ids()

    for i in image_ids:
        print(i)
        caps = cap_loader.get_captions(i)
        tmp = [pos_tag(word_tokenize(str(a).lower())) for a in caps]
        pos.append(tmp)

    pos = [p3 for p1 in pos for p2 in p1 for p3 in p2]
    pos = [(w, '<UNK>') if mapping.get(t) is None else (w, mapping[t]) for (w, t) in pos]
    vocab = Counter(elem for elem in pos)
    vocab = vocab.most_common()

    word = [w for ((w, t), c) in vocab]
    word = [w for w in word if w not in stop_words]
    pos = [t for ((w, t), c) in vocab]
    count = [c for ((_, _), c) in vocab]

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

    no_punctuation = [i for (i, x) in enumerate(words) if x not in punctuations]
    words = [words[i] for i in no_punctuation]
    counts = [counts[i] for i in no_punctuation]
    poss = [poss[i] for i in no_punctuation]

    vocab = {'words': words, 'counts': counts, 'poss': poss}
    return vocab


def build_dictionary():
    mapping = {'NNS': 'NN', 'NNP': 'NN', 'NNPS': 'NN', 'NN': 'NN',
               'VB': 'VB', 'VBD': 'VB', 'VBN': 'VB', 'VBZ': 'VB', 'VBP': 'VB', 'VBG': 'VB',
               'JJR': 'JJ', 'JJS': 'JJ', 'JJ': 'JJ',
               'DT': 'DT', 'PRP': 'PRP', 'PRP$': 'PRP', 'IN': 'IN'}

    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    vocab = get_vocab(punctuations, mapping)
    vocab = get_top_k_vocab(vocab, config.VOCAB_SIZE)
    save_variables(config.DICTIONARY_PATH, vocab)


if __name__ == "__main__":
    if not os.path.exists(config.DICTIONARY_PATH):
        build_dictionary()

    saved_vocab = load_variables(config.DICTIONARY_PATH)
    print(saved_vocab['words'])
    print(saved_vocab['counts'])
