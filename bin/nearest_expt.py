#!/usr/bin/python

import numpy as np
from r222.wordvectors import WordVectors
import r222.utils as ru
import itertools
import logging

logging.basicConfig(filename="output/nearest_expt.log", filemode="w", level=logging.INFO, format="%(message)s")

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
NEAREST_INPUT_FILE_PATH = "aux/nearest_input.txt"

CONJ1_FILE_PATH = "data/conj1.txt"
CONJ2_FILE_PATH = "data/conj2.txt"
CONJ3_COUNTRIES_FILE_PATH = "data/conj3countries.txt"
CONJ3_SPORTS_FILE_PATH = "data/conj3sports.txt"
CONJ3_ANIMALS_FILE_PATH = "data/conj3animals.txt"
CONJ3_OCCUPATIONS_FILE_PATH = "data/conj3occupations.txt"
CONJ4_FILE_PATH = "data/conj4.txt"

NUM_SPLITS = 10

K = 10

nearest_input = ru.read_set(NEAREST_INPUT_FILE_PATH)

s_1, n_1 = ru.read_sn(CONJ1_FILE_PATH)
s_2, n_2 = ru.read_sn(CONJ2_FILE_PATH)
s_3_countries, n_3_countries = ru.read_sn(CONJ3_COUNTRIES_FILE_PATH)
s_3_sports, n_3_sports = ru.read_sn(CONJ3_SPORTS_FILE_PATH)
s_3_animals, n_3_animals = ru.read_sn(CONJ3_ANIMALS_FILE_PATH)
s_3_occupations, n_3_occupations = ru.read_sn(CONJ3_OCCUPATIONS_FILE_PATH)

conj1 = ru.conj(s_1, n_1)
conj2 = ru.conj(s_2, n_2)
conj3_countries = ru.conj(s_3_countries, n_3_countries)
conj3_sports = ru.conj(s_3_sports, n_3_sports)
conj3_animals = ru.conj(s_3_animals, n_3_animals)
conj3_occupations = ru.conj(s_3_occupations, n_3_occupations)

def nearest(f, label, k):
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

logging.info("Running nearest (add)")
nearest(np.add, "add", K)

logging.info("Running nearest (mult)")
nearest(np.multiply, "mult", K)

logging.info("Running nearest (conj1)")
nearest(ru.f_conj(conj1), "conj1", K)

logging.info("Running nearest (conj2)")
nearest(ru.f_conj(conj2), "conj2", K)
