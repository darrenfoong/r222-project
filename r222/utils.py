import numpy as np
import math
from scipy.optimize import minimize
import logging

def cos_sim(u, v):
    return abs(np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v)))

def centroid_vector(vectors):
    return np.mean(vectors, axis=0)

def sum_cos_sim(vector, vectors):
    norm_vectors = np.linalg.norm(vectors, axis=1)
    norm_vectors *= np.linalg.norm(vector)
    prod = np.dot(vector, np.transpose(vectors))
    prod /= norm_vectors
    return np.sum(np.abs(prod))

def sum_cos_sim_curry(vectors):
    def f(vector):
        return -1.0 * sum_cos_sim(vector, vectors)
    return f

def con_ortho(ortho):
    def f(vector):
        return np.dot(ortho, vector)
    return f

def con_ortho_jac(ortho):
    def f(vector):
        return ortho
    return f

def furthest_vector(ortho, vectors):
    x0 = np.ones(300)
    constraints = { 'type': 'eq', 'fun': con_ortho(ortho), 'jac': con_ortho_jac(ortho) }
    options = { 'disp': True }

    def callback(x):
        logging.info(str(sum_cos_sim(x, vectors)))

    res = minimize(sum_cos_sim_curry(vectors), x0, constraints=constraints, options=options, callback=callback)

    return res.x

def read_sn(conj_file_path):
    with open(conj_file_path, "r") as conj_file:
        lines = conj_file.readlines()
        lines = map((lambda line: line.split(" ")), lines)

        s = map((lambda s: float(s)), lines[0])
        n = map((lambda s: float(s)), lines[1])

    # np.newaxis necessary for successful np.tranpose() in conj()
    return np.array(s)[np.newaxis], np.array(n)[np.newaxis]

def write_sn(conj_file_path, s, n):
    with open(conj_file_path, "w") as conj_file:
        s_str = " ".join(map((lambda x: str(x)), s))
        n_str = " ".join(map((lambda x: str(x)), n))
        conj_file.write(s_str + "\n")
        conj_file.write(n_str + "\n")

def read_set(set_file_path):
    res = set()

    with open(set_file_path, "r") as set_file:
        for line in iter(set_file):
            line = line.replace("\n", "")
            res.add(line)

    return res

def split_set(ans, num_splits):
    size = len(ans)
    ans_list = list(ans)
    size_split = int(math.ceil(size/float(num_splits)))
    return [ans_list[x:x+size_split] for x in range(0, size, size_split)]

def best_sn(ans, word_vectors):
    return np.ones(300), np.ones(300)

def conj(s, n):
    return np.multiply(np.transpose(np.kron(s, s)), s) + np.multiply(np.transpose(np.kron(s, n)), n) + np.multiply(np.transpose(np.kron(n, s)), n) + np.multiply(np.transpose(np.kron(n, n)), n)
