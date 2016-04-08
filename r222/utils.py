import numpy as np
import math
from scipy.optimize import minimize
import logging

# I/O

def read_sn(conj_file_path):
    with open(conj_file_path, "r") as conj_file:
        lines = conj_file.readlines()
        lines = map((lambda line: line.split(" ")), lines)

        s = map((lambda s: float(s)), lines[0])
        n = map((lambda s: float(s)), lines[1])

    return np.array(s), np.array(n)

def write_sn(conj_file_path, s, n):
    with open(conj_file_path, "w") as conj_file:
        s_str = " ".join(map((lambda x: str(x)), s))
        n_str = " ".join(map((lambda x: str(x)), n))
        conj_file.write(s_str + "\n")
        conj_file.write(n_str + "\n")

def read_set(set_file_path, split=False):
    res = set()

    with open(set_file_path, "r") as set_file:
        for line in iter(set_file):
            line = line.replace("\n", "")

            if split:
                res.add(tuple(line.split(" ")))
            else:
                res.add(line)

    return res

def split_set(ans, num_splits):
    size = len(ans)
    ans_list = list(ans)
    size_split = int(math.ceil(size/float(num_splits)))
    return [ans_list[x:x+size_split] for x in range(0, size, size_split)]

# Linear algebra

def cos_sim(u, v):
    return abs(np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v)))

def centroid_vector(vectors):
    return np.mean(vectors, axis=0)

def big_cos_sim(vector, vectors):
    norm_vectors = np.linalg.norm(vectors, axis=1)
    norm_vectors *= np.linalg.norm(vector)
    prod = np.dot(vector, np.transpose(vectors))
    prod /= norm_vectors
    return prod

def sum_cos_sim(vector, vectors):
    prod = big_cos_sim(vector, vectors)
    return np.sum(np.abs(prod))

def sum_cos_sim_curry(vectors):
    def f(vector):
        return -1.0 * sum_cos_sim(vector, vectors)
    return f

def sum_cos_sim2(sn, avs, nvs, anvs):
    res = 0.0

    s = sn[0:300]
    n = sn[300:600]

    c = conj(s, n)

    anvs_c = big_dotkron2(avs, nvs, c)

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

def dotkron(a, n, c):
    # original dotkron
    return np.dot(np.kron(a, n), c)

def dotkron2(a, n, c):
    # dotkron with einsum, same performance
    cp = np.reshape(c, (300, 300, 300))
    return np.einsum("i,j,ijk->k", a, n, cp)

def big_kron(avs, nvs):
    rows = np.shape(avs)[0]
    columns = np.shape(avs)[1]
    columnsq = columns * columns

    res = np.empty(shape=(rows, columnsq))

    for i in range(0, rows):
        res[i] = np.kron(avs[i], nvs[i])

    return res

def big_dotkron(avs, nvs, c):
    # original big_dotkron with big_kron, fastest
    # but uses most memory because of intermediate an_kron
    an_kron = big_kron(avs, nvs)
    return np.dot(an_kron, c)

def big_dotkron2(avs, nvs, c):
    # slower big_dotkron
    # but uses less memory because no intermediate
    cp = np.reshape(c, (300, 300, 300))
    return np.einsum("li,lj,ijk->lk", avs, nvs, cp)

def big_dotkron3(avs, nvs, c):
    # slower than 1, faster than 2
    # doesn't use less memory
    cp = np.reshape(c, (300, 300, 300))
    inter = np.einsum("lj,ijk->lik", nvs, cp)
    return np.einsum("li,lik->lk", avs, inter)

def big_dotkron4(avs, nvs, c):
    # slowest of all, despite using tensordot
    # doesn't use less memory
    cp = np.reshape(c, (300, 300, 300))
    inter = np.tensordot(nvs, cp, axes=([1],[1]))
    return np.einsum("li,lik->lk", avs, inter)

def conj(s, n):
    return np.outer(np.kron(s, s), s) + np.outer(np.kron(s, n), n) + np.outer(np.kron(n, s), n) + np.outer(np.kron(n, n), n)

def f_conj(conj):
    def f(av, nv):
        return dotkron(av, nv, conj)
    return f

# Optimisation

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

def nearest_vectors(embedding, word_vectors, k):
    embeddings = word_vectors._embeddings
    cos_sim = big_cos_sim(embedding, embeddings)
    cos_sim_sorted = np.argsort(cos_sim)

    return map(word_vectors.rget, cos_sim_sorted[:k])
