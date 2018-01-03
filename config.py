import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

TAGS_PATH = os.path.join(ROOT_PATH, 'datasets/tags.txt')
CAPTIONS_PATH = os.path.join(ROOT_PATH, 'datasets/captions.token')
DICTIONARY_PATH = os.path.join(ROOT_PATH, 'datasets/dictionary.pkl')

FEATURES_PATH = os.path.join(ROOT_PATH, 'datasets/features.npz')
LABELS_PATH = os.path.join(ROOT_PATH, 'datasets/labels.npy')

TEST_IMAGES_DIR = os.path.join(ROOT_PATH, 'images')

NUM_TERMS = 6
VOCAB_SIZE = 500
POS_TAGS = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBG', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}
