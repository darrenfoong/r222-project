#!/usr/bin/python

import numpy as np
import math
from r222.wordvectors import WordVectors
import r222.utils as ru
import itertools
import logging

logging.basicConfig(filename="output/gen_space.log", filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
ADJECTIVES_FILE_PATH = "data/adjectives.txt"
NOUNS_FILE_PATH = "data/nouns.txt"

CONJ1_FILE_PATH = "data/conj1.txt"
CONJ2_FILE_PATH = "data/conj2.txt"

adjectives = list(ru.read_set(ADJECTIVES_FILE_PATH))
nouns = list(ru.read_set(NOUNS_FILE_PATH))

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

    nvs = np.empty(shape=(len(nouns), 300))

    for i in range(len(nouns)):
        nvs[i] = word_vectors.get(nouns[i])

    for adjective in adjectives:
        logging.info("Processing adjective " + str(count))

        av = word_vectors.get(adjective)
        res = f(av, nvs)

        for i in range(len(nouns)):
            key = "YYY_" + adjective + "_" + nouns[i] + "_YYY"
            embedding = res[i]
            word_vectors._add(key, embedding)

        count += 1

    word_vectors.serialize(VECTORS_FILE_PATH + "." + label)

logging.info("Running gen_space (add)")
gen_space(np.add, "add")

logging.info("Running gen_space (mult)")
gen_space(np.multiply, "mult")

logging.info("Running gen_space (conj1)")
gen_space(ru.f_conj(conj1), "conj1")

logging.info("Running gen_space (conj2)")
gen_space(ru.f_conj(conj2), "conj2")
