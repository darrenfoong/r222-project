#!/usr/bin/python

VECTORS_FILE_PATH = "data/vectors-sk-an.lemmas.50-min-count.15-iters"
ADJECTIVES_FILE_PATH = "data/adjectives.txt"
NOUNS_FILE_PATH = "data/nouns.txt"

adjectives = set()
nouns = set()

with open(VECTORS_FILE_PATH, "r") as vectors_file, \
     open(ADJECTIVES_FILE_PATH, "w") as adjectives_file, \
     open(NOUNS_FILE_PATH, "w") as nouns_file:

     for line in iter(vectors_file):
         entry = line.split(" ")[0]
         if entry.startswith("XXX_"):
             entry_words = entry.split("_")
             adjectives.add(entry_words[1])
             nouns.add(entry_words[2])

     print str(len(adjectives)) + " unique adjectives"
     print str(len(nouns)) + " unique nouns"

     for adjective in sorted(adjectives):
         adjectives_file.write(adjective + "\n")

     for noun in sorted(nouns):
         nouns_file.write(noun + "\n")
