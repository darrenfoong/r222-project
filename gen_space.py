#!/usr/bin/python

import numpy as np
import itertools

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
ADJECTIVES_FILE_PATH = "data/adjectives.txt"
NOUNS_FILE_PATH = "data/nouns.txt"
AN_FILE_PATH = "data/an.txt"

class Vectors:
    def __init__(self, size):
        self.size = size
        self._index = dict()
        self._vectors = np.empty(shape=(size, 300))

    def add(self, word, vector):
        current_index = len(self._index)
        self._index[word] = current_index
        self._vectors[current_index] = vector/np.linalg.norm(vector)

    def get(self, word):
        if word in self._index:
            return self._vectors[self._index[word]]
        else:
            return None

with open(VECTORS_FILE_PATH, "r") as vectors_file:
    num_words = sum(1 for line in vectors_file)
    print "Number of words: " + str(num_words)

with open(AN_FILE_PATH, "r") as an_file:
    num_ans = sum(1 for line in an_file)
    print "Number of AN pairs: " + str(num_ans)

input_vectors = Vectors(num_words)
adjectives = list()
nouns = list()

with open(VECTORS_FILE_PATH, "r") as vectors_file:

    for line in iter(vectors_file):
        line_split = line.split(" ")
        word_vector = line_split[1:-1]
        input_vectors.add(line_split[0], map((lambda s: float(s)), word_vector))

    print "Input vectors: " + str(input_vectors.size)

with open(ADJECTIVES_FILE_PATH, "r") as adjectives_file, \
     open(NOUNS_FILE_PATH, "r") as nouns_file:

    adjectives = adjectives_file.read().split("\n")[:-1]
    nouns = nouns_file.read().split("\n")[:-1]

    print "Adjectives: " + str(len(adjectives))
    print "Nouns: " + str(len(nouns))

num_ans_all = len(adjectives) * len(nouns)

#an_vectors_add = Vectors(num_ans_all)
#an_vectors_mult = Vectors(num_ans_all)

def cos_sim(u, v):
    return abs(np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v)))

an_count = 0
cuml_add_sim = 0.0
cuml_mult_sim = 0.0

with open(AN_FILE_PATH, "r") as an_file:
    for line in iter(an_file):
        line = line[:-1]
        line_split = line.split(" ")
        adjective = line_split[0]
        noun = line_split[1]

        adjective_noun = adjective + "_" + noun
        an_vector = input_vectors.get("XXX_" + adjective_noun + "_XXX")

        if an_vector is None:
            continue

        adjective_vector = input_vectors.get(adjective)
        noun_vector = input_vectors.get(noun)

        if adjective_vector is None:
            continue

        if noun_vector is None:
            continue

        an_vector_add = np.add(adjective_vector, noun_vector)
        an_vector_mult = np.multiply(adjective_vector, noun_vector)

        an_count += 1
        cuml_add_sim += cos_sim(an_vector, an_vector_add)
        cuml_mult_sim += cos_sim(an_vector, an_vector_mult)

print "Total AN pairs: " + str(an_count)
print "Average cosine similarity (addition): " + str(cuml_add_sim/an_count)
print "Average cosine similarity (multiplication): " + str(cuml_mult_sim/an_count)
