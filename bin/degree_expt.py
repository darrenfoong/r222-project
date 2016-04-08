#!/usr/bin/python

import numpy as np
from r222.wordvectors import WordVectors
import r222.utils as ru
import itertools
import logging

logging.basicConfig(filename="output/degree_expt.log", filemode="w", level=logging.INFO, format="%(message)s")

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"

CONJ1_FILE_PATH = "data/conj1.txt"
CONJ2_FILE_PATH = "data/conj2.txt"
CONJ3_COUNTRIES_FILE_PATH = "data/conj3countries.txt"
CONJ3_SPORTS_FILE_PATH = "data/conj3sports.txt"
CONJ3_ANIMALS_FILE_PATH = "data/conj3animals.txt"
CONJ3_OCCUPATIONS_FILE_PATH = "data/conj3occupations.txt"
CONJ4_FILE_PATH = "data/conj4.txt"

DEGREE_INPUT_FILE_PATH = "data/degree_input.txt"

NUM_SPLITS = 10

word_vectors = WordVectors(VECTORS_FILE_PATH, 300, "UNKNOWN")

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

degree_input = ru.read_set(DEGREE_INPUT_FILE_PATH)

for line in nearest_input:
    line_split = line.split(";")
    line_split_split = line_split[0].split(" ")
    adjective = line_split_split[0]
    noun = line_split_split[1]
    target = line_split[1]

    av = word_vectors.get(adjective)
    nv = word_vectors.get(noun)
    tv = word_vectors.get(target)

    cos_sim1 = ru.cos_sim(ru.dotkron(av, nv, conj1), tv)
    cos_sim2 = ru.cos_sim(ru.dotkron(av, nv, conj2), tv)

    logging.info(line + " = " + str(cos_sim1) + " ; " + str(cos_sim2))
