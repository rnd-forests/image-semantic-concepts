import re
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
                self.caps[img_id].append(cap)
                line = file.readline()

    def get_img_ids(self):
        return list(self.caps.keys())

    def get_captions(self, img_id):
        return self.caps[img_id]

    def get_all_captions(self):
        return self.caps
