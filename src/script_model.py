import dataset
import pprint
import script_map
import pickle
import itertools
from math import log, sqrt
from collections import defaultdict
import vector
from data_gen import segmentation


def extract_sentences(data, group):
    for doc in data:
        for l, st in dataset.iter_sentences(doc):
            if l in group:
                yield l, st

def extract_all_sentences(data):
    for doc in data:
        for l, st in dataset.iter_sentences(doc):
            yield l, st


def load_dataset():
    data = list(dataset.read_dataset('weighted'))

    train = list(extract_all_sentences(data[:16000]))
    dev = list(extract_all_sentences(data[16000:18000]))
    test = list(extract_all_sentences(data[18000:]))

    return train, dev, test


def entropy(seq):
    t = tuple(seq)
    s = sum(t)
    return -sum((i/s) * log(i/s) for i in t)


def entropy_map(entropies, dist):
    #return dist
    return {k: v / entropies.get(k, 1) for k, v in dist.items()}


def n_grams(st, n):
    for i in range(len(st)-n+1):
        yield st[i:i+n]


def substrings(word):
    padded = ' ' + word + ' '
    for i in range(len(padded)):
        for j in range(i+2, len(padded)+1):
            yield padded[i:j]


def iterate_words(data, group):
    for doc in data:
        for l, word in dataset.iter_words(doc):
            if l not in group or script_map.script(word[0]) == 'Common':
                continue
            yield l, word

def iterate_words_st(sentences):
    for l, st in sentences:
        for word in segmentation.by_words(st):
            if script_map.script(word[0]) == 'Common':
                continue
            yield l, word


def find_most_similar(vec1, lang_vectors):
    cosine_gen = ((l2, vec1.cosine(vec2)) \
                  for l2, vec2 in lang_vectors.items())
    max_kv = max(cosine_gen, key=lambda x: x[1])
    return max_kv


# calculate lang to lang matrix
def build_similarity_matrix(lang_vectors):
    lang_similarity = defaultdict(dict)

    for l1, v1 in lang_vectors.items():
        for l2, v2 in lang_vectors.items():
            similarity = v1.cosine(v2)
            lang_similarity[l1][l2] = similarity

    return lang_similarity


def estimate_language(seq, lang_vectors, vec_ctor):
    dist = defaultdict(int)
    for item in seq:
        dist[item] += 1
    vec = vec_ctor(dist)
    return find_most_similar(vec, lang_vectors)


def build_result_matrix(data, identifier):
    result_matrix = defaultdict(lambda: defaultdict(int))
    correct = 0
    total = 0
    for l1, unit in data:
        l2, similarity = identifier(unit)
        result_matrix[l1][l2] += 1
        total += 1
        if l1 == l2:
            correct += 1
    print(correct/total)
    return result_matrix


def print_result_matrix(matrix):
    langs = sorted(matrix.keys())
    for l1 in langs:
        dist = matrix[l1]
        total = sum(dist.values())
        correct = dist[l1]
        accuracy = correct / total
        print(l1, accuracy)
        if accuracy < 0.9:
            print(sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5])


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
    merged_vec = vec1.add(vec2)
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


def cluster_languages(lang_vectors, result_matrix, threshold):
    group_vectors = {frozenset({l}): vec for l, vec in lang_vectors.items()}
    group_map = {l: frozenset({l}) for l in lang_vectors.keys()}

    while True:
        lang_similarity = build_similarity_matrix(group_vectors)
        l1, l2 = max(iter_group_pair(group_vectors), \
                     key=lambda x: lang_similarity[x[0]][x[1]])
        if lang_similarity[l1][l2] < threshold:
            break

        merge(group_vectors, group_map, l1, l2)
        expected_error_rate = \
            calc_expected_error_rate(group_map, result_matrix)

        print(lang_similarity[l1][l2], expected_error_rate)
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
    cluster, lang_vectors, scripts = train_script_level(train)
    #print_script_cluster(cluster, lang_vectors, scripts)

    #for g in cluster:
    #    if len(g) == 1:
    #        # don't need to go into furthermore
    #        continue
    #
    #    entropies, lang_vectors = train_word_level(g, train)
    #    # TODO: train_sentence

    save_model(cluster, lang_vectors, scripts)


def train_script_level(data):
    # count no. of scripts and scripts chars
    scripts, scripts_per_lang = script_level_statistics(data)
    vec_ctor = vector.NPVector.constructor(scripts)

    # calculate a vector for each language
    lang_vectors = {l: vec_ctor(dist) \
                    for l, dist in scripts_per_lang.items()}

    # calculate confusion matrix
    result_matrix = build_result_matrix(data, lambda st:
        estimate_language(script_map.script_str(st), lang_vectors, vec_ctor))

    # using confusion matrix, cluster languages
    cluster = cluster_languages(lang_vectors, result_matrix, 0.98)

    return cluster, lang_vectors, scripts


def script_level_statistics(data):
    scripts = set()
    lang_scr_cnt = defaultdict(lambda: defaultdict(int))

    for lang, sentence in data:
        for ch in sentence:
            scr = script_map.script(ch)

            scripts.add(scr)
            lang_scr_cnt[lang][scr] += 1

    return sorted(scripts), lang_scr_cnt



def train_sentence_level(train, dev):
    word_lang, lang_word = word_level_statstics(train)
    entropies, lang_vectors = build_lang_vectors(word_lang, lang_word, 0.001)

    def find_most_similar(st):
        d = defaultdict(float)
        for w in st:
            if script_map.script(w[0]) == 'Common':
                continue
            if w not in word_lang:
                # TODO : use word identifier
                continue
            for k, v in word_lang[w].items():
                d[k] += v / entropies.get(w, 1)

        if len(d) == 0:
            #print('wtf')
            return ' ', 0.0

        return max(d.items(), key=lambda x: x[1])

    def iterate(data):
        for l, st in data:
            yield l, segmentation.by_words(st)

    result_matrix = build_result_matrix(iterate(dev), find_most_similar)
    print_result_matrix(result_matrix)

    #for k in total.keys():
    #    if correct[k] / total[k] < 0.9:
    #        print(result_matrix[k])
    #    print(k, correct[k], total[k], correct[k] / total[k])

    return entropies, lang_vectors



def word_level_statstics(train):
    word_lang = defaultdict(lambda: defaultdict(int))
    lang_word = defaultdict(lambda: defaultdict(int))

    for l, word in iterate_words_st(train):
        word_lang[word][l] += 1
        lang_word[l][word] += 1

    return word_lang, lang_word


def build_lang_vectors(item_lang, lang_item, constant):
    entropies = {i: entropy(l.values()) + constant \
                 for i, l in item_lang.items()}
    lang_vectors = {l: vector.SparseVector(entropy_map(entropies, dist)) \
                    for l, dist in lang_item.items()}
    return entropies, lang_vectors


def train_word_level(train):
    substr_lang, lang_substr = char_level_statistics(train)
    entropies, lang_vectors = build_lang_vectors(substr_lang, lang_substr, 0.1)

    return entropies, lang_vectors


def char_level_statistics(train):
    substr_lang = defaultdict(lambda: defaultdict(int))
    lang_substr = defaultdict(lambda: defaultdict(int))

    for l, word in iterate_words_st(train):
        for s in substrings(word):
            substr_lang[s][l] += 1
            lang_substr[l][s] += 1

    return substr_lang, lang_substr


def doit3(train, dev):
    entropies, lang_vectors = train_word_level(train)

    def build_vector(w):
        dist = defaultdict(int)
        for s in substrings(w):
            dist[s] += 1
        return vector.SparseVector(entropy_map(entropies, dist))

    def find_most_similar(w):
        vec = build_vector(w)
        cosine_gen = ((l2, vec.cosine(vec2)) \
                      for l2, vec2 in lang_vectors.items())
        max_kv = sorted(cosine_gen, key=lambda x: x[1], reverse=True)
        return max_kv[0]

    result_matrix = build_result_matrix( \
        iterate_words_st(dev), find_most_similar)


def split_data_by_group(data, cluster):
    group_map = {}
    for g in cluster:
        for l in g:
            group_map[l] = g

    split_data = defaultdict(list)
    for l, st in data:
        g = group_map[l]
        split_data[g].append((l, st))
    return split_data


def classify():
    train, dev, test = load_dataset()
    cluster, lang_vectors, scripts = load_model()
    split_train = split_data_by_group(train, cluster)
    split_dev = split_data_by_group(dev, cluster)
    for g in cluster:
        t = split_train[g]
        d = split_dev[g]
        if len(g) == 1:
            # identification is finished here for those languages
            # TODO: tag it
            continue
        print(g)
        #entropies, lang_vectors = train_word_level(t)
        #
        doit3(t, d)
        #train_sentence_level(t, d)

        #print(g, len(st))


if __name__ == '__main__':
    train()
    #classify()


'''
# document classification
for doc in dev:
    for l1, sentence in dataset.iter_sentences(doc):
        dist = defaultdict(int)
        for scr in script_map.script_str(sentence):
            dist[scr] += 1
        vec = dist_to_vector(dist, scripts)

        l2, similarity = find_most_similar(vec1)


def doit_(group, data, name):
    word_cnt = defaultdict(lambda: defaultdict(int))
    for doc in data:
        for l, word in dataset.iter_words(doc):
            if l not in group or script_map.script(word[0]) == 'Common':
                continue
            word_cnt[word][l] += 1

    dist = defaultdict(int)
    dist2 = defaultdict(int)
    for w, langs in word_cnt.items():
        dist[len(langs)] += 1
        dist2[frozenset(langs)] += 1

    print(sorted(dist.items(), key=lambda x:x[0]))
    print(sorted(dist2.items(), key=lambda x:x[1]))


def doit4(group, data, dev):
    a = defaultdict(int)
    b = defaultdict(lambda: defaultdict(int))
    c = defaultdict(lambda: defaultdict(int))
    train_words = [(l, w) for l, w in words_(data, group)]
    test_words = [(l, w) for l, w in words_(dev, group)]
    for l, w in train_words:
        for s in substrings(w):
            a[s] += 1
            b[l][s] += 1
            c[s][l] += 1

    def entropy_map(dist):
        #return dist
        return {k: v / entropies.get(k, 1) for k, v in dist.items()}

    entropies = {w: entropy(l.values())+0.01 for w, l in c.items()}

    train_set = []
    test_set = []

    for l, w in train_words:
        dist = defaultdict(int)
        for s in substrings(w):
            dist[s] += 1
        train_set.append((dist, l))
    for l, w in test_words:
        dist = defaultdict(int)
        for s in substrings(w):
            dist[s] += 1
        test_set.append(dist)

    classifier = MaxentClassifier.train(train_set)
    result = classifier.classify_many(test_set)
    correct = 0
    total = 0
    for i, j in zip(result, test_words):
        if i == j[0]:
            correct += 1
        total += 1
    print(correct, total, correct/total)

    #for pdist in classifier.prob_classify_many(test):
    #    print('%.4f %.4f' % (pdist.prob('x'), pdist.prob('y')))
    #print(train_set[0], test_set[0])

'''

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