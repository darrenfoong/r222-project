import numpy as np
import itertools
import logging

class WordVectors:
    def __init__(self, path, size_embeddings, unk_string, extra=0):
        self._unk = 0
        self._unk_string = unk_string

        self._map = dict()
        self._rmap = dict()
        self._size_embeddings = size_embeddings

        with open(path, "r") as embeddings_file:
            num_embeddings = sum(1 for line in embeddings_file) + extra

        self._embeddings = np.empty(shape=(num_embeddings, self._size_embeddings))

        with open(path, "r") as embeddings_file:
            for line in iter(embeddings_file):
                line = line.replace("\n", "")
                line_split = line.split(" ")[:-1]
                embedding = line_split[1:]
                self._add(line_split[0], map((lambda s: float(s)), embedding))

    def serialize(self, path):
        with open(path, "w", buffering=512*(1024**2)) as embeddings_file:
            for key, index in self._map.iteritems():
                embedding = self._embeddings[index]
                output = " ".join(map((lambda x: str(x)), embedding))
                embeddings_file.write(key + " " + output + " \n")

    def get(self, key):
        if key in self._map:
            return self._embeddings[self._map[key]]
        else:
            return self._embeddings[self._unk]

    def rget(self, index):
        if index in self._rmap:
            return self._rmap[index]
        else:
            return None

    def _add(self, key, embedding):
        current_index = len(self._map)
        self._map[key] = current_index
        self._rmap[current_index] = key
        self._embeddings[current_index] = embedding/np.linalg.norm(embedding)

        if key == self._unk_string:
            self._unk = current_index
            logging.info("wordVectors has previous UNK: " + key)
            logging.info("Remapping UNK")

    def filter(self, path):
        with open(path, "r") as filter_file:
            num_filter = sum(1 for line in filter_file)

        res = np.empty(shape=(num_filter, self._size_embeddings))
        current_index = 0

        with open(path, "r") as filter_file:
            for line in iter(filter_file):
                line = line.replace("\n", "")
                res[current_index] = self.get(line)
                current_index += 1

        return res
