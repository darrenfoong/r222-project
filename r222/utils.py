import numpy as np
import logging

def cos_sim(u, v):
    return abs(np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v)))

def centroid_vector(vectors):
    return np.mean(vectors, axis=0)

def furthest_vector(ortho, vectors):
    return ortho

def read_sn(conj_file_path):
    with open(conj_file_path, "r") as conj_file:
        lines = conj_file.readlines()
        lines = map((lambda line: line.split(" ")), lines)

        s = map((lambda s: float(s)), lines[0])
        n = map((lambda s: float(s)), lines[1])

    # np.newaxis necessary for successful np.tranpose() in conj()
    return np.array(s)[np.newaxis], np.array(n)[np.newaxis]

def conj(s, n):
    return np.multiply(np.transpose(np.kron(s, s)), s) + np.multiply(np.transpose(np.kron(s, n)), n) + np.multiply(np.transpose(np.kron(n, s)), n) + np.multiply(np.transpose(np.kron(n, n)), n)
