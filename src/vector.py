import numpy as np
from math import sqrt

class NPVector:
    def __init__(self, vec, features):
        self.vec = vec
        # precompute an absolute value
        self.abs = sqrt(np.square(vec).sum())
        self.features = features

    @classmethod
    def fromdist(cls, dist, features):
        vec = np.array([dist.get(scr, 0) for scr in features])
        return cls(vec, features)

    def constructor(features):
        return lambda x: NPVector.fromdist(x, features)

    def add(self, rhs):
        assert self.features == rhs.features
        return NPVector(self.vec + rhs.vec, self.features)

    def cosine(self, rhs):
        assert self.features == rhs.features
        dot = np.dot(self.vec, rhs.vec)
        return dot / (self.abs * rhs.abs)


class SparseVector:
    def __init__(self, dist):
        self.dist = dist
        # precompute an absolute value
        self.abs = sqrt(sum(i * i for i in dist.values()))

    @classmethod
    def fromdist(cls, dist):
        return cls(dist)

    def add(self, rhs):
        newdist = {}
        newdist.update(self.dist)
        for k, v in rhs.dist.items():
            newdist[k] = newdist.get(k, 0) + v
        return SparseVector(newdist)

    def cosine(self, rhs):
        if len(self.dist) > len(rhs.dist):
            s, l = rhs.dist, self.dist
        else:
            s, l = self.dist, rhs.dist

        dot = sum(l.get(k, 0) * v for k, v in s.items())
        return dot / (self.abs * rhs.abs)

