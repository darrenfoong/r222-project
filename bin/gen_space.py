#!/usr/bin/python

import numpy as np
import math
from r222.wordvectors import WordVectors
import r222.utils as ru
import itertools
import logging

logging.basicConfig(filename="output/gen_space.log", filemode="w", level=logging.INFO, format="%(message)s")

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
ADJECTIVES_FILE_PATH = "data/adjectives.txt"
NOUNS_FILE_PATH = "data/nouns.txt"

CONJ1_FILE_PATH = "data/conj1.txt"
CONJ2_FILE_PATH = "data/conj2.txt"

adjectives = ru.read_set(ADJECTIVES_FILE_PATH)
nouns = ru.read_set(NOUNS_FILE_PATH)

logging.info("Adjectives: " + str(len(adjectives)))
logging.info("Nouns: " + str(len(nouns)))

s_1, n_1 = ru.read_sn(CONJ1_FILE_PATH)
s_2, n_2 = ru.read_sn(CONJ2_FILE_PATH)

conj1 = ru.conj(s_1, n_1)
conj2 = ru.conj(s_2, n_2)

def gen_space(f, label):
    size = len(adjectives)*len(nouns)

    word_vectors = WordVectors(VECTORS_FILE_PATH, 300, "UNKNOWN", extra=size)

    count = 1
    count_freq = max(10**(math.floor(math.log10(size))) / 100, 1)

    for adjective, noun in itertools.product(adjectives, nouns):
        key = "YYY_" + adjective + "_" + noun + "_YYY"
        embedding = f(word_vectors.get(adjective), word_vectors.get(noun))
        word_vectors._add(key, embedding)

        count += 1
        if count % count_freq == 0:
            logging.info("Processed " + label + "/" + str(count))

    word_vectors.serialize(VECTORS_FILE_PATH + "." + label)

logging.info("Running gen_space (add)")
gen_space(np.add, "add")

logging.info("Running gen_space (mult)")
gen_space(np.multiply, "mult")

logging.info("Running gen_space (conj1)")
gen_space(ru.f_conj(conj1), "conj1")

logging.info("Running gen_space (conj2)")
gen_space(ru.f_conj(conj2), "conj2")
