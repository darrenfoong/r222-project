#!/usr/bin/python

import numpy as np
from r222.wordvectors import WordVectors
import r222.utils as ru
import logging

logging.basicConfig(filename="output/nearest_expt.log", filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
NEAREST_INPUT_FILE_PATH = "aux/nearest_input.txt"

CONJ1_FILE_PATH = "data/conj1.txt"
CONJ2_FILE_PATH = "data/conj2.txt"

K = 10

s_1, n_1 = ru.read_sn(CONJ1_FILE_PATH)
s_2, n_2 = ru.read_sn(CONJ2_FILE_PATH)

conj1 = ru.conj(s_1, n_1)
conj2 = ru.conj(s_2, n_2)

nearest_input = ru.read_set(NEAREST_INPUT_FILE_PATH)

def nearest(f, label, k, s=None, n=None):
    word_vectors = WordVectors(VECTORS_FILE_PATH + "." + label, 300, "UNKNOWN")

    for line in nearest_input:
        line_split = line.split(" ")

        if len(line_split) == 1:
            word = line_split[0]
            embedding = word_vectors.get(word)
        else:
            adjective = line_split[0]
            noun = line_split[1]
            embedding = f(word_vectors.get(adjective), word_vectors.get(noun))

        nearest = ru.nearest_vectors(embedding, word_vectors, k)

        logging.info("Nearest neighbours for " + line + ": ")
        logging.info(nearest)

    if s:
        nearest = ru.nearest_vectors(s, word_vectors, k)

        logging.info("Nearest neighbours for true vector: ")
        logging.info(nearest)

    if n:
        nearest = ru.nearest_vectors(n, word_vectors, k)

        logging.info("Nearest neighbours for false vector: ")
        logging.info(nearest)

logging.info("Running nearest (add)")
nearest(np.add, "add", K)

logging.info("Running nearest (mult)")
nearest(np.multiply, "mult", K)

logging.info("Running nearest (conj1)")
nearest(ru.f_conj(conj1), "conj1", K, s_1, n_1)

logging.info("Running nearest (conj2)")
nearest(ru.f_conj(conj2), "conj2", K, s_2, n_2)
