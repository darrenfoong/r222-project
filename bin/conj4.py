#!/usr/bin/python

import numpy as np
from r222.wordvectors import WordVectors
import r222.utils as ru
import logging

logging.basicConfig(filename="output/conj4.log", filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
AN_FILE_PATH = "data/an.txt"

CONJ4_FILE_PATH = "data/conj4.txt"

NUM_SPLITS = 10

word_vectors = WordVectors(VECTORS_FILE_PATH, 300, "UNKNOWN")

ans = ru.read_set(AN_FILE_PATH)
ans_split = ru.split_set(ans, NUM_SPLITS)

for i in range(0, NUM_SPLITS):
    ans_split_training = ans_split[0:i] + ans_split[i+1:NUM_SPLITS]
    ans_training = [item for sublist in ans_split_training for item in sublist]
    logging.info("AN split " + str(i+1) + "; " + str(len(ans_split[i])) + " and " + str(len(ans_training)) + " elements")
    s, n = ru.best_sn(ans_training, word_vectors)
    ru.write_sn(CONJ4_FILE_PATH + "." + str(i+1), s, n)

logging.info("Done")
