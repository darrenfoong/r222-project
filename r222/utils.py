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

def big_kron(avs, nvs):
    rows = np.shape(avs)[0]
    columns = np.shape(avs)[1]
    columnsq = columns * columns

    res = np.empty(shape=(rows, columnsq))

    for i in range(0, rows):
        res[i] = np.kron(avs[i], nvs[i])

    return res

def sum_cos_sim2(sn, avs, nvs, anvs):
    res = 0.0

    s = sn[0:300][np.newaxis]
    n = sn[300:600][np.newaxis]

    c = conj(s, n)

    an_kron = big_kron(avs, nvs)
    anvs_c = np.dot(an_kron, c)

    prod_inter = np.multiply(anvs_c, anvs)
    prod = np.sum(prod_inter, axis=1)

    anvs_norms = np.linalg.norm(anvs, axis=1)
    anvs_c_norms = np.linalg.norm(anvs_c, axis=1)

    norms = np.multiply(anvs_norms, anvs_c_norms)
    prod /= norms

    return np.sum(np.abs(prod))

def sum_cos_sim2_curry(avs, nvs, anvs):
    def f(sn):
        return sum_cos_sim2(sn, avs, nvs, anvs)
    return f

def con_ortho2(sn):
    return np.dot(sn[0:300], sn[300:600])

def best_sn(ans, word_vectors):
    sn0 = np.ones(600)
    constraints = { 'type': 'eq', 'fun': con_ortho2 }
    options = { 'disp': True }

    num_ans = len(ans)

    avs = np.empty(shape=(num_ans, 300))
    nvs = np.empty(shape=(num_ans, 300))
    anvs = np.empty(shape=(num_ans, 300))
    current_index = 0

    for an in ans:
        line_split = an.split(" ")
        adjective = line_split[0]
        noun = line_split[1]

        adjective_noun = adjective + "_" + noun
        anvs[current_index] = word_vectors.get("XXX_" + adjective_noun + "_XXX")

        avs[current_index] = word_vectors.get(adjective)
        nvs[current_index] = word_vectors.get(noun)

        current_index += 1

    logging.info("Pre-processing done")

    def callback(sn):
        logging.info(str(sum_cos_sim2(sn, avs, nvs, anvs)))

    res = minimize(sum_cos_sim2_curry(avs, nvs, anvs), sn0, constraints=constraints, options=options, callback=callback)

    sn_split = np.split(res.x, 2, axis=0)

    return sn_split[0], sn_split[1]

def conj(s, n):
    return np.dot(np.transpose(np.kron(s, s)), s) + np.dot(np.transpose(np.kron(s, n)), n) + np.dot(np.transpose(np.kron(n, s)), n) + np.dot(np.transpose(np.kron(n, n)), n)
