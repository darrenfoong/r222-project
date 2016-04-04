#!/usr/bin/python

import numpy as np
from r222.wordvectors import WordVectors
import r222.utils
import logging

logging.basicConfig(filename="output/conj2.log", filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
CONJ2_FILE_PATH = "data/conj2.txt"

word_vectors = WordVectors(VECTORS_FILE_PATH, 300, "UNKNOWN")

s_2 = r222.utils.centroid_vector(word_vectors._embeddings)
n_2 = r222.utils.furthest_vector(s_2, word_vectors._embeddings)

with open(CONJ2_FILE_PATH, "w") as conj2_file:
    s_2_str = " ".join(map((lambda x: str(x)), s_2))
    n_2_str = " ".join(map((lambda x: str(x)), n_2))
    conj2_file.write(s_2_str + "\n")
    conj2_file.write(n_2_str + "\n")

logging.info("Done")
