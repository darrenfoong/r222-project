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
CONJ3_COUNTRIES_FILE_PATH = "data/conj3countries.txt"
CONJ3_SPORTS_FILE_PATH = "data/conj3sports.txt"
CONJ3_ANIMALS_FILE_PATH = "data/conj3animals.txt"
CONJ3_OCCUPATIONS_FILE_PATH = "data/conj3occupations.txt"
CONJ4_FILE_PATH = "data/conj4.txt"

NUM_SPLITS = 10

K = 10

adjectives = ru.read_set(ADJECTIVES_FILE_PATH)
nouns = ru.read_set(NOUNS_FILE_PATH)

logging.info("Adjectives: " + str(len(adjectives)))
logging.info("Nouns: " + str(len(nouns)))

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

def gen_space(f, label):
    size = len(adjectives)*len(nouns)

    word_vectors = WordVectors(VECTORS_FILE_PATH, 300, "UNKNOWN", extra=size)

    count = 1
    count_freq = max(10**(math.floor(math.log10(size))) / 1000, 1)

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
