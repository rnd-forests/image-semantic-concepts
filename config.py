import os

TAGS_COUNT = 6
VOCAB_SIZE = 500

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

TAGS_FILE = os.path.join(ROOT_PATH, 'datasets/tags.txt')
CAPTIONS_FILE = os.path.join(ROOT_PATH, 'datasets/captions.token')
VOCABULARY_FILE = os.path.join(ROOT_PATH, 'datasets/vocabulary.pkl')
FEATURES_FILE = os.path.join(ROOT_PATH, 'datasets/features.npz')
LABELS_FILE = os.path.join(ROOT_PATH, 'datasets/labels.npy')
TEST_IMAGES_DIR = os.path.join(ROOT_PATH, 'images')
