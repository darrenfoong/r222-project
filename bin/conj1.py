#!/usr/bin/python

import numpy as np
from r222.wordvectors import WordVectors
import r222.utils
import logging

logging.basicConfig(filename="output/conj1.log", filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
CONJ1_FILE_PATH = "data/conj1.txt"

word_vectors = WordVectors(VECTORS_FILE_PATH, 300, "UNKNOWN")

s_1 = np.ones(300)
n_1 = r222.utils.furthest_vector(s_1, word_vectors._embeddings)

with open(CONJ1_FILE_PATH, "w") as conj1_file:
    s_1_str = " ".join(map((lambda x: str(x)), s_1))
    n_1_str = " ".join(map((lambda x: str(x)), n_1))
    conj1_file.write(s_1_str + "\n")
    conj1_file.write(n_1_str + "\n")

logging.info("Done")
