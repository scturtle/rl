import random

class ReplayQueue:
    def __init__(self, limit):
        self.f = []
        self.r = []
        self.cnt = 0
        self.limit = limit

    def push(self, item):
        if self.cnt == self.limit:
            self.pop()
        self.cnt += 1
        self.r.append(item)

    def pop(self):
        if not self.f:
            self.f, self.r = self.r, []
            self.f.reverse()
        self.cnt -= 1
        return self.f.pop()

    def size(self):
        return len(self.f) + len(self.r)

    def sample(self, n):
        fc = len(self.f)
        rc = len(self.r)
        fn = int(round(1.0 * n * fc / (fc + rc)))
        return random.sample(self.f, fn) + random.sample(self.r, n - fn)
