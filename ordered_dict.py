from collections import OrderedDict


class DefaultListOrderedDict(OrderedDict):
    def __missing__(self, key):
        self[key] = []
        return self[key]
