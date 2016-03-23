#!/usr/bin/python

import numpy as np

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
ADJECTIVES_FILE_PATH = "data/adjectives.txt"
NOUNS_FILE_PATH = "data/nouns.txt"

with open(VECTORS_FILE_PATH, "r") as vectors_file:
    num_words = sum(1 for line in vectors_file)
    print "Number of words: " + str(num_words)

word_index = dict()
word_vectors = np.empty(shape=(num_words, 300))

with open(VECTORS_FILE_PATH, "r") as vectors_file, \
     open(ADJECTIVES_FILE_PATH, "r") as adjectives_file, \
     open(NOUNS_FILE_PATH, "r") as nouns_file:

    for line in iter(vectors_file):
        line_split = line.split(" ")
        current_index = len(word_index)
        word_index[line_split[0]] = current_index
        word_vector = line_split[1:-1]
        word_vectors[current_index] = map((lambda s: float(s)), word_vector)

    print "Word index: " + str(len(word_index))
