#!/usr/bin/python

import numpy as np
from r222.wordvectors import WordVectors
import r222.utils as ru
import logging

logging.basicConfig(filename="output/conj3.log", filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
COUNTRIES_FILE_PATH = "aux/countries.txt"
SPORTS_FILE_PATH = "aux/sports.txt"
ANIMALS_FILE_PATH = "aux/animals.txt"
OCCUPATIONS_FILE_PATH = "aux/occupations.txt"

CONJ3_COUNTRIES_FILE_PATH = "data/conj3countries.txt"
CONJ3_SPORTS_FILE_PATH = "data/conj3sports.txt"
CONJ3_ANIMALS_FILE_PATH = "data/conj3animals.txt"
CONJ3_OCCUPATIONS_FILE_PATH = "data/conj3occupations.txt"

word_vectors = WordVectors(VECTORS_FILE_PATH, 300, "UNKNOWN")

countries_embeddings = word_vectors.filter(COUNTRIES_FILE_PATH)
sports_embeddings = word_vectors.filter(SPORTS_FILE_PATH)
animals_embeddings = word_vectors.filter(ANIMALS_FILE_PATH)
occupations_embeddings = word_vectors.filter(OCCUPATIONS_FILE_PATH)

s_3_countries = ru.normalize(ru.centroid_vector(countries_embeddings))
n_3_countries = ru.normalize(ru.furthest_vector(s_3_countries, countries_embeddings))

ru.write_sn(CONJ3_COUNTRIES_FILE_PATH, s_3_countries, n_3_countries)

s_3_sports = ru.normalize(ru.centroid_vector(sports_embeddings))
n_3_sports = ru.normalize(ru.furthest_vector(s_3_sports, sports_embeddings))

ru.write_sn(CONJ3_SPORTS_FILE_PATH, s_3_sports, n_3_sports)

s_3_animals = ru.normalize(ru.centroid_vector(animals_embeddings))
n_3_animals = ru.normalize(ru.furthest_vector(s_3_animals, animals_embeddings))

ru.write_sn(CONJ3_ANIMALS_FILE_PATH, s_3_animals, n_3_animals)

s_3_occupations = ru.normalize(ru.centroid_vector(occupations_embeddings))
n_3_occupations = ru.normalize(ru.furthest_vector(s_3_occupations, occupations_embeddings))

ru.write_sn(CONJ3_OCCUPATIONS_FILE_PATH, s_3_occupations, n_3_occupations)

logging.info("Done")
