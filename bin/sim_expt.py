#!/usr/bin/python

import numpy as np
from r222.wordvectors import WordVectors
import r222.utils
import itertools
import logging

logging.basicConfig(filename="output/sim_expt.log", filemode="w", level=logging.INFO, format="%(message)s")

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
ADJECTIVES_FILE_PATH = "data/adjectives.txt"
NOUNS_FILE_PATH = "data/nouns.txt"
AN_FILE_PATH = "data/an.txt"

CONJ1_FILE_PATH = "data/conj1.txt"
CONJ2_FILE_PATH = "data/conj2.txt"

word_vectors = WordVectors(VECTORS_FILE_PATH, 300, "UNKNOWN")
adjectives = list()
nouns = list()

logging.info("Word vectors: " + str(len(word_vectors._map)))

with open(ADJECTIVES_FILE_PATH, "r") as adjectives_file, \
     open(NOUNS_FILE_PATH, "r") as nouns_file:

    adjectives = adjectives_file.read().split("\n")[:-1]
    nouns = nouns_file.read().split("\n")[:-1]

    logging.info("Adjectives: " + str(len(adjectives)))
    logging.info("Nouns: " + str(len(nouns)))

an_count = 0
cuml_add_sim = 0.0
cuml_mult_sim = 0.0
cuml_conj1_sim = 0.0
cuml_conj2_sim = 0.0

s_1, n_1 = r222.utils.read_sn(CONJ1_FILE_PATH)
s_2, n_2 = r222.utils.read_sn(CONJ2_FILE_PATH)

conj1 = r222.utils.conj(s_1, n_1)
conj2 = r222.utils.conj(s_2, n_2)

with open(AN_FILE_PATH, "r") as an_file:
    for line in iter(an_file):
        line = line[:-1]
        line_split = line.split(" ")
        adjective = line_split[0]
        noun = line_split[1]

        adjective_noun = adjective + "_" + noun
        an_vector = word_vectors.get("XXX_" + adjective_noun + "_XXX")

        if an_vector is None:
            continue

        adjective_vector = word_vectors.get(adjective)
        noun_vector = word_vectors.get(noun)

        if adjective_vector is None:
            continue

        if noun_vector is None:
            continue

        logging.info("AN " + str(an_count+1))

        an_vector_add = np.add(adjective_vector, noun_vector)
        an_vector_mult = np.multiply(adjective_vector, noun_vector)
        an_vector_conj1 = np.dot(np.kron(adjective_vector, noun_vector), conj1)
        an_vector_conj2 = np.dot(np.kron(adjective_vector, noun_vector), conj2)

        an_count += 1
        cuml_add_sim += r222.utils.cos_sim(an_vector, an_vector_add)
        cuml_mult_sim += r222.utils.cos_sim(an_vector, an_vector_mult)
        cuml_conj1_sim += r222.utils.cos_sim(an_vector, an_vector_conj1)
        cuml_conj2_sim += r222.utils.cos_sim(an_vector, an_vector_conj2)

logging.info("Total AN pairs: " + str(an_count))
logging.info("Average cosine similarity (addition): " + str(cuml_add_sim/an_count))
logging.info("Average cosine similarity (multiplication): " + str(cuml_mult_sim/an_count))
logging.info("Average cosine similarity (conj1): " + str(cuml_conj1_sim/an_count))
logging.info("Average cosine similarity (conj2): " + str(cuml_conj2_sim/an_count))
