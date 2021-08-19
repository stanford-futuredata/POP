# For computing fib entries
from itertools import count


class Counter(object):
    def __init__(self):
        self.counter = count()
        self.paths_dict = {}

    def __getitem__(self, path):
        if not isinstance(path, tuple):
            path = tuple(path)
        if path not in self.paths_dict:
            self.paths_dict[path] = next(self.counter)

        return self.paths_dict[path]
