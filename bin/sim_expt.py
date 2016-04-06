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

COUNTRIES_FILE_PATH = "aux/countries.txt"
SPORTS_FILE_PATH = "aux/sports.txt"
ANIMALS_FILE_PATH = "aux/animals.txt"
OCCUPATIONS_FILE_PATH = "aux/occupations.txt"

CONJ1_FILE_PATH = "data/conj1.txt"
CONJ2_FILE_PATH = "data/conj2.txt"
CONJ3_COUNTRIES_FILE_PATH = "data/conj3countries.txt"
CONJ3_SPORTS_FILE_PATH = "data/conj3sports.txt"
CONJ3_ANIMALS_FILE_PATH = "data/conj3animals.txt"
CONJ3_OCCUPATIONS_FILE_PATH = "data/conj3occupations.txt"
CONJ4_FILE_PATH = "data/conj4.txt"

NUM_SPLITS = 10

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

countries = r222.utils.read_set(COUNTRIES_FILE_PATH)
sports = r222.utils.read_set(SPORTS_FILE_PATH)

an_count = 0
an_countries_count = 0
an_sports_count = 0
an_animals_count = 0
an_occupations_count = 0
cuml_add_sim = 0.0
cuml_mult_sim = 0.0
cuml_conj1_sim = 0.0
cuml_conj2_sim = 0.0
cuml_conj3_countries_sim = 0.0
cuml_conj3_sports_sim = 0.0
cuml_conj3_animals_sim = 0.0
cuml_conj3_occupations_sim = 0.0

s_1, n_1 = r222.utils.read_sn(CONJ1_FILE_PATH)
s_2, n_2 = r222.utils.read_sn(CONJ2_FILE_PATH)
s_3_countries, n_3_countries = r222.utils.read_sn(CONJ3_COUNTRIES_FILE_PATH)
s_3_sports, n_3_sports = r222.utils.read_sn(CONJ3_SPORTS_FILE_PATH)
s_3_animals, n_3_animals = r222.utils.read_sn(CONJ3_ANIMALS_FILE_PATH)
s_3_occupations, n_3_occupations = r222.utils.read_sn(CONJ3_OCCUPATIONS_FILE_PATH)

conj1 = r222.utils.conj(s_1, n_1)
conj2 = r222.utils.conj(s_2, n_2)
conj3_countries = r222.utils.conj(s_3_countries, n_3_countries)
conj3_sports = r222.utils.conj(s_3_sports, n_3_sports)
conj3_animals = r222.utils.conj(s_3_animals, n_3_animals)
conj3_occupations = r222.utils.conj(s_3_occupations, n_3_occupations)

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

        cuml_add_sim += r222.utils.cos_sim(an_vector, an_vector_add)
        cuml_mult_sim += r222.utils.cos_sim(an_vector, an_vector_mult)
        cuml_conj1_sim += r222.utils.cos_sim(an_vector, an_vector_conj1)
        cuml_conj2_sim += r222.utils.cos_sim(an_vector, an_vector_conj2)

        an_count += 1

        if noun in countries:
            an_vector_conj3_countries = np.dot(np.kron(adjective_vector, noun_vector), conj3_countries)
            cuml_conj3_countries_sim += r222.utils.cos_sim(an_vector, an_vector_conj3_countries)
            an_countries_count += 1

        if noun in sports:
            an_vector_conj3_sports = np.dot(np.kron(adjective_vector, noun_vector), conj3_sports)
            cuml_conj3_sports_sim += r222.utils.cos_sim(an_vector, an_vector_conj3_sports)
            an_sports_count += 1

        if noun in animals:
            an_vector_conj3_animals = np.dot(np.kron(adjective_vector, noun_vector), conj3_animals)
            cuml_conj3_animals_sim += r222.utils.cos_sim(an_vector, an_vector_conj3_animals)
            an_animals_count += 1

        if noun in occupations:
            an_vector_conj3_occupations = np.dot(np.kron(adjective_vector, noun_vector), conj3_occupations)
            cuml_conj3_occupations_sim += r222.utils.cos_sim(an_vector, an_vector_conj3_occupations)
            an_occupations_count += 1

logging.info("Total AN pairs: " + str(an_count))
logging.info("Average cosine similarity (addition): " + str(cuml_add_sim/an_count) + " (" + str(an_count) + ")")
logging.info("Average cosine similarity (multiplication): " + str(cuml_mult_sim/an_count) + " (" + str(an_count) + ")")
logging.info("Average cosine similarity (conj1): " + str(cuml_conj1_sim/an_count) + " (" + str(an_count) + ")")
logging.info("Average cosine similarity (conj2): " + str(cuml_conj2_sim/an_count) + " (" + str(an_count) + ")")

if an_countries_count == 0:
    logging.info("an_countries_count = 0")
else:
    logging.info("Average cosine similarity (conj3countries): " + str(cuml_conj3_countries_sim/an_countries_count) + " (" + str(an_countries_count) + ")")

if an_sports_count == 0:
    logging.info("an_sports_count = 0")
else:
    logging.info("Average cosine similarity (conj3sports): " + str(cuml_conj3_sports_sim/an_sports_count) + " (" + str(an_sports_count) + ")")

if an_animals_count == 0:
    logging.info("an_animals_count = 0")
else:
    logging.info("Average cosine similarity (conj3animals): " + str(cuml_conj3_animals_sim/an_animals_count) + " (" + str(an_animals_count) + ")")

if an_occupations_count == 0:
    logging.info("an_occupations_count = 0")
else:
    logging.info("Average cosine similarity (conj3occupations): " + str(cuml_conj3_occupations_sim/an_occupations_count) + " (" + str(an_occupations_count) + ")")

ans = r222.utils.read_set(AN_FILE_PATH)
ans_split = r222.utils.split_set(ans, NUM_SPLITS)

cuml_conj4_sims = list()

for i in range(0, NUM_SPLITS):
    s_4, n_4 = r222.utils.read_sn(CONJ4_FILE_PATH + "." + str(i+1))
    conj4 = r222.utils.conj(s_4, n_4)

    an_count = 0
    cuml_conj4_sim = 0.0

    for line in ans_split[i]:
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

        an_vector_conj4 = np.dot(np.kron(adjective_vector, noun_vector), conj4)

        cuml_conj4_sim += r222.utils.cos_sim(an_vector, an_vector_conj4)

        an_count += 1

    avg = cuml_conj4_sim/an_count

    logging.info("Average cosine similarity (conj4/" + str(i+1) + "): " + str(avg) + " (" + str(an_count) + ")")

    cuml_conj4_sims.append(avg)

logging.info("Average cosine similarity (conj4): " + str(np.mean(cuml_conj4_sims)))
