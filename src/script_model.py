import dataset
import pprint
import script_map
import numpy as np
import pickle
from math import log, sqrt
from collections import defaultdict


def load_dataset():
    data = list(dataset.read_dataset('132_langs'))
    train = data[8000:]
    dev = data[1000:]
    test = data[1000:]
    return train, dev, test


def collect_statistics(data):
    scripts = set()
    lang_scr_cnt = defaultdict(lambda: defaultdict(int))

    for doc in data:
        for lang, sentence in dataset.iter_sentences(doc):
            for ch in sentence:
                scr = script_map.script(ch)

                scripts.add(scr)
                lang_scr_cnt[lang][scr] += 1

    return sorted(scripts), lang_scr_cnt


# cosine similarity
def dist_to_vector(dist, features):
    vector = np.array([dist.get(scr, 0) for scr in features])
    abs_dist = sqrt(np.square(vector).sum()) # precompute an absolute value
    return vector, abs_dist


# don't need to be centroid calculation, using cosine similarity normailizes it
def add_vectors(v1, v2):
    vector = v1[0] + v2[0]
    abs_dist = sqrt(np.square(vector).sum())
    return vector, abs_dist


def cosine(a, b):
    abs_a = a[1]
    abs_b = b[1]
    dot = np.dot(a[0], b[0])

    return dot / (abs_a * abs_b)


def find_most_similar(vec1, lang_vectors):
    cosine_gen = ((l2, cosine(vec1, vec2)) \
                  for l2, vec2 in lang_vectors.items())
    max_kv = max(cosine_gen, key=lambda x: x[1])
    return max_kv


# calculate lang to lang matrix
def calc_similarity_matrix(lang_vectors):
    lang_similarity = defaultdict(dict)

    for l1, v1 in lang_vectors.items():
        for l2, v2 in lang_vectors.items():
            similarity = cosine(v1, v2)
            lang_similarity[l1][l2] = similarity

    return lang_similarity


def estimate_language(doc, lang_vectors, scripts):
    dist = defaultdict(int)
    for scr in script_map.script_str(doc):
        dist[scr] += 1
    vec = dist_to_vector(dist, scripts)
    return find_most_similar(vec, lang_vectors)


def calc_result_matrix(lang_vectors, data, scripts):
    result_matrix = defaultdict(lambda: defaultdict(int))
    for doc in data:
        for l1, sentence in dataset.iter_sentences(doc):
            l2, similarity = estimate_language(sentence, lang_vectors, scripts)
            result_matrix[l1][l2] += 1
    return result_matrix


def calc_expected_error_rate(groups, result_matrix):
    correct = 0
    error = 0
    for l1, dist in result_matrix.items():
        for l2, cnt in dist.items():
            if l2 in groups[l1]:
                correct += cnt
            else:
                error += cnt
    return error / (correct + error)


# Agglomerative clustering
def merge(vectors, groups, set1, set2):
    vec1 = vectors.pop(set1)
    vec2 = vectors.pop(set2)
    merged_vec = add_vectors(vec1, vec2)
    merged_set = set1 | set2
    vectors[merged_set] = merged_vec
    for i in merged_set:
        groups[i] = merged_set


def iter_group_pair(group_vectors):
    groups = list(group_vectors.keys())
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]
            yield g1, g2


def cluster_languages(lang_vectors, result_matrix):
    group_vectors = {frozenset({l}): vec for l, vec in lang_vectors.items()}
    group_map = {l: frozenset({l}) for l in lang_vectors.keys()}

    while True:
        lang_similarity = calc_similarity_matrix(group_vectors)
        l1, l2 = max(iter_group_pair(group_vectors), \
                     key=lambda x: lang_similarity[x[0]][x[1]])
        if lang_similarity[l1][l2] < 0.98:
            break

        merge(group_vectors, group_map, l1, l2)
        expected_error_rate = \
            calc_expected_error_rate(group_map, result_matrix)

        if expected_error_rate < 0.01:
            break

    return tuple(group_vectors.keys())


def print_script_cluster(cluster, lang_scr_cnt):
    for g in cluster:
        print('----------------')
        for l in g:
            s = sum(lang_scr_cnt[l].values())
            m = max(lang_scr_cnt[l].items(), key=lambda x: x[1])
            print('{}\t{}'.format(l, m[0]))


def save_model(cluster, lang_vectors, scripts):
    with open('script_model.pkl', 'wb') as f:
        pickle.dump(cluster, f)
        pickle.dump(lang_vectors, f)
        pickle.dump(scripts, f)


def load_model():
    with open('script_model.pkl', 'rb') as f:
        cluster = pickle.load(f)
        lang_vectors = pickle.load(f)
        scripts = pickle.load(f)
        return cluster, lang_vectors, scripts


def train():
    train, dev, test = load_dataset()

    # count no. of scripts and scripts chars
    scripts, lang_scr_cnt = collect_statistics(train)

    # calculate a vector for each language
    lang_vectors = {l: dist_to_vector(dist, scripts) \
                    for l, dist in lang_scr_cnt.items()}

    # calculate confusion matrix
    result_matrix = calc_result_matrix(lang_vectors, train, scripts)

    # using confusion matrix, cluster languages
    cluster = cluster_languages(lang_vectors, result_matrix)

    print_script_cluster(cluster, lang_scr_cnt)

    save_model(cluster, lang_vectors, scripts)


def classify_documents(lang_vectors, data, scripts, cluster):
    result = defaultdict(list)
    for doc in data:
        for l1, sentence in dataset.iter_sentences(doc):
            # TODO: sometime multilingual sentence come up,
            #       maybe fallback method for low similarity doc can be used?
            l2, _ = estimate_language(sentence, lang_vectors, scripts)
            group = next(g for g in cluster if l2 in g)
            result[group].append((l1, sentence))
    return result


def cnt(docs):
    a = defaultdict(int)
    b = defaultdict(lambda: defaultdict(int))
    for l, d in docs:
        for ch in d:
            a[ch] += 1
            b[l][ch] += 1
    s = sum(a.values())
    a = { k: v for k, v in a.items() if v > 10 }
    pprint.pprint(a)
    #pprint.pprint(b)



def classify():
    train, dev, test = load_dataset()

    cluster, lang_vectors, scripts = load_model()
    result = classify_documents(lang_vectors, dev, scripts, cluster)
    for g, st in result.items():
        if len(g) == 1:
            pass  # identification is finished here

        if len(g) > 20:
            cnt(st)

        #print(g, len(st))


if __name__ == '__main__':
    train()
    #classify()


#def iter_lang_pairs():
#    for k, v in lang_similarity.items():


#for l1 in scr_vecs.keys():
#    for l2 in scr_vecs.keys():
#        print(l1, l2, lang_similarity[l1][l2], lang_matrix[l1][l2])
'''
# document classification
for doc in dev:
    for l1, sentence in dataset.iter_sentences(doc):
        dist = defaultdict(int)
        for scr in script_map.script_str(sentence):
            dist[scr] += 1
        vec = dist_to_vector(dist, scripts)

        l2, similarity = find_most_similar(vec1)
'''