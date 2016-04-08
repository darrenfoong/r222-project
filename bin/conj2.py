#!/usr/bin/python

import numpy as np
from r222.wordvectors import WordVectors
import r222.utils as ru
import logging

logging.basicConfig(filename="output/conj2.log", filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
CONJ2_FILE_PATH = "data/conj2.txt"

word_vectors = WordVectors(VECTORS_FILE_PATH, 300, "UNKNOWN")

s_2 = ru.centroid_vector(word_vectors._embeddings)
n_2 = ru.furthest_vector(s_2, word_vectors._embeddings)

ru.write_sn(CONJ2_FILE_PATH, s_2, n_2)

logging.info("Done")
