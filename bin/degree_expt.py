#!/usr/bin/python

from r222.wordvectors import WordVectors
import r222.utils as ru
import logging

logging.basicConfig(filename="output/degree_expt.log", filemode="w", level=logging.INFO, format="%(message)s")

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
DEGREE_INPUT_FILE_PATH = "aux/degree_input.txt"

CONJ1_FILE_PATH = "data/conj1.txt"
CONJ2_FILE_PATH = "data/conj2.txt"
CONJ3_COUNTRIES_FILE_PATH = "data/conj3countries.txt"
CONJ3_SPORTS_FILE_PATH = "data/conj3sports.txt"
CONJ3_ANIMALS_FILE_PATH = "data/conj3animals.txt"
CONJ3_OCCUPATIONS_FILE_PATH = "data/conj3occupations.txt"

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

for line in degree_input:
    line_split = line.split(";")
    line_split_split = line_split[0].split(" ")
    adjective = line_split_split[0]
    noun = line_split_split[1]
    target = line_split[1]

    av = word_vectors.get(adjective)
    nv = word_vectors.get(noun)
    tv = word_vectors.get(target)

    dk1 = ru.dotkron(av, nv, conj1)
    dk2 = ru.dotkron(av, nv, conj2)
    dk3_countries = ru.dotkron(av, nv, conj3_countries)
    dk3_sports = ru.dotkron(av, nv, conj3_sports)
    dk3_animals = ru.dotkron(av, nv, conj3_animals)
    dk3_occupations = ru.dotkron(av, nv, conj3_occupations)

    cos_sim1 = ru.cos_sim(dk1, tv)
    cos_sim2 = ru.cos_sim(dk2, tv)
    cos_sim3_countries = ru.cos_sim(dk3_countries, tv)
    cos_sim3_sports = ru.cos_sim(dk3_sports, tv)
    cos_sim3_animals = ru.cos_sim(dk3_animals, tv)
    cos_sim3_occupations = ru.cos_sim(dk3_occupations, tv)

    cos_sim1_s = ru.cos_sim(dk1, s_1)
    cos_sim2_s = ru.cos_sim(dk2, s_2)
    cos_sim3_countries_s = ru.cos_sim(dk3_countries, s_3_countries)
    cos_sim3_sports_s = ru.cos_sim(dk3_sports, s_3_sports)
    cos_sim3_animals_s = ru.cos_sim(dk3_animals, s_3_animals)
    cos_sim3_occupations_s = ru.cos_sim(dk3_occupations, s_3_occupations)

    logging.info(line)
    logging.info(" conj1: " + str(cos_sim1) + " (" + str(cos_sim1_s) + ")")
    logging.info(" conj2: " + str(cos_sim2) + " (" + str(cos_sim2_s) + ")")
    logging.info(" conj3countries: " + str(cos_sim3_countries) + " (" + str(cos_sim3_countries_s) + ")")
    logging.info(" conj3sports: " + str(cos_sim3_sports) + " (" + str(cos_sim3_sports_s) + ")")
    logging.info(" conj3animals: " + str(cos_sim3_animals) + " (" + str(cos_sim3_animals_s) + ")")
    logging.info(" conj3occupations: " + str(cos_sim3_occupations) + " (" + str(cos_sim3_occupations_s) + ")")
