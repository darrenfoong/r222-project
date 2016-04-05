#!/usr/bin/python

import numpy as np
from r222.wordvectors import WordVectors
import r222.utils
import logging

logging.basicConfig(filename="output/conj3.log", filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
COUNTRIES_FILE_PATH = "data/countries.txt"
SPORTS_FILE_PATH = "data/sports.txt"

CONJ3_COUNTRIES_FILE_PATH = "data/conj3countries.txt"
CONJ3_SPORTS_FILE_PATH = "data/conj3sports.txt"

word_vectors = WordVectors(VECTORS_FILE_PATH, 300, "UNKNOWN")

countries_embeddings = word_vectors.filter(COUNTRIES_FILE_PATH)
sports_embeddings = word_vectors.filter(SPORTS_FILE_PATH)

s_3_countries = r222.utils.centroid_vector(countries_embeddings)
n_3_countries = r222.utils.furthest_vector(s_3_countries, countries_embeddings)

r222.utils.write_sn(CONJ3_COUNTRIES_FILE_PATH, s_3_countries, n_3_countries)

s_3_sports = r222.utils.centroid_vector(sports_embeddings)
n_3_sports = r222.utils.furthest_vector(s_3_sports, sports_embeddings)

r222.utils.write_sn(CONJ3_SPORTS_FILE_PATH, s_3_sports, n_3_sports)

logging.info("Done")
