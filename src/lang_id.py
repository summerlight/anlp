import dataset
import pprint
import script_map
import pickle
import itertools
import argparse
from math import log, sqrt
from collections import defaultdict
import vector
from data_gen import segmentation

def default_int():
    return defaultdict(int)


word_entropy_coefficient = 0.1
sentence_entropy_coefficient = 0.001


class ScriptLangID:
    def __init__(self):
        self.scripts = set()
        self.sorted_scripts = []
        self.scripts_per_lang = defaultdict(default_int)
        self.lang_vectors = {}

    def train(self, data):
        self.collect_stats(data)
        self.lang_vectors = \
            {l: vector.NPVector.fromdist(dist, self.sorted_scripts) \
                for l, dist in self.scripts_per_lang.items()}

    def identify(self, word):
        return self.most_similar(word)[0]

    def identify_many(self, data):
        return [(self.most_similar(word)[0], word) for word in data]

    def collect_stats(self, data):
        for lang, sentence in data:
            for ch in sentence:
                scr = script_map.script(ch)

                self.scripts.add(scr)
                self.scripts_per_lang[lang][scr] += 1

        self.sorted_scripts = sorted(self.scripts)

    def most_similar(self, st):
        vec = self.build_vector(st)
        cosine_gen = ((l2, vec.cosine(vec2)) \
                      for l2, vec2 in self.lang_vectors.items())
        max_kv = sorted(cosine_gen, key=lambda x: x[1], reverse=True)
        return max_kv[0]

    def build_vector(self, st):
        dist = defaultdict(int)
        for scr in script_map.script_str(st):
            dist[scr] += 1
        return vector.NPVector.fromdist(dist, self.sorted_scripts)


class WordLangID:
    def __init__(self, entropy_constant):
        self.substr_lang = defaultdict(default_int)
        self.lang_substr = defaultdict(default_int)
        self.ent_const = entropy_constant
        self.entropies = {}
        self.lang_vectors = {}

    def train(self, data):
        self.collect_stats(data)
        self.entropies = {i: entropy(l.values()) + self.ent_const \
                          for i, l in self.substr_lang.items()}
        self.lang_vectors = { \
            l: vector.SparseVector(entropy_map(self.entropies, dist)) \
            for l, dist in self.lang_substr.items() \
        }

    def identify(self, word):
        return self.most_similar(word)[0][0]

    def identify_many(self, data):
        return [(self.most_similar(word)[0][0], word) for word in data]

    def identify_detail(self, word):
        return self.most_similar(word)[:5]

    def collect_stats(self, data):
        for l, word in data:
            for s in substrings(word):
                self.substr_lang[s][l] += 1
                self.lang_substr[l][s] += 1

    def most_similar(self, word):
        vec = self.build_vector(word)
        cosine_gen = ((l2, vec.cosine(vec2)) \
                      for l2, vec2 in self.lang_vectors.items())
        max_kv = sorted(cosine_gen, key=lambda x: x[1], reverse=True)
        return max_kv

    def build_vector(self, word):
        dist = defaultdict(int)
        for s in substrings(word):
            dist[s] += 1
        return vector.SparseVector(entropy_map(self.entropies, dist))


class SentenceLangID:
    def __init__(self, entropy_constant, word_langid):
        self.word_langid = word_langid
        self.word_lang = defaultdict(default_int)
        self.lang_word = defaultdict(default_int)
        self.entropies = {}
        self.ent_const = entropy_constant
        self.lang_vectors = {}

    def train(self, data):
        self.collect_stats(data)
        self.entropies = {i: entropy(l.values()) + self.ent_const \
                          for i, l in self.word_lang.items()}
        self.lang_vectors = { \
            l: vector.SparseVector(entropy_map(self.entropies, dist)) \
            for l, dist in self.lang_word.items() \
        }

    def identify(self, st):
        return self.most_similar(st)[0]

    def identify_many(self, data):
        a = sorted(self.entropies.items(), key=lambda x: x[1])
        return [(self.most_similar(st)[0], st) for st in data]

    def collect_stats(self, train):
        for l, word in iterate_words_st(train):
            self.word_lang[word][l] += 1
            self.lang_word[l][word] += 1

    def most_similar(self, st):
        d = defaultdict(float)
        for w in segmentation.by_words(st):
            if script_map.script(w[0]) == 'Common':
                continue
            if w not in self.word_lang:
                for k, v in self.unknown_word_score(w):
                #    pass
                    d[k] += v
                continue
            for k, v in self.word_lang[w].items():
                d[k] += v #/ self.entropies.get(w, 1)

        if len(d) == 0:
            return ' ', 0.0

        return max(d.items(), key=lambda x: x[1])

    def unknown_word_score(self, w):
        result = self.word_langid.identify_detail(w)
        entropy_val = entropy(i[1] for i in result if i[1] > 0)
        s = sum(i[1] for i in result) * 0.2

        for i in result:
            if i[1] > 0:
                yield i[0], i[1] #/ s*(entropy_val + 0.01)


def load_dataset():
    data = list(dataset.read_dataset('weighted'))

    train = list(extract_all_sentences(data[:16000]))
    dev = list(extract_all_sentences(data[16000:18000]))
    test = list(extract_all_sentences(data[18000:]))

    return train, dev, test


def save_model(file_path, cluster, script_id, st_langid_map):
    with open(file_path, 'wb') as f:
        pickle.dump(cluster, f)
        pickle.dump(script_id, f)
        pickle.dump(st_langid_map, f)


def load_model(file_path):
    with open(file_path, 'rb') as f:
        cluster = pickle.load(f)
        script_id = pickle.load(f)
        st_langid_map = pickle.load(f)
        return cluster, script_id, st_langid_map


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


def iterate_words_st(sentences):
    for l, st in sentences:
        for word in segmentation.by_words(st):
            if script_map.script(word[0]) == 'Common':
                continue
            yield l, word


def extract_all_sentences(data):
    for doc in data:
        for l, st in dataset.iter_sentences(doc):
            yield l, st


# calculate lang to lang matrix
def build_similarity_matrix(lang_vectors):
    lang_similarity = defaultdict(dict)

    for l1, v1 in lang_vectors.items():
        for l2, v2 in lang_vectors.items():
            similarity = v1.cosine(v2)
            lang_similarity[l1][l2] = similarity

    return lang_similarity


def build_result_matrix(actual, result):
    result_matrix = defaultdict(lambda: defaultdict(int))
    correct = 0
    total = 0
    for i1, i2 in zip(actual, result):
        l1, _ = i1
        l2, _ = i2
        result_matrix[l1][l2] += 1
        total += 1
        if l1 == l2:
            correct += 1
    #print(correct/total)
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

        #print(lang_similarity[l1][l2], expected_error_rate)
        if expected_error_rate < 0.01:
            break

    return tuple(group_vectors.keys())


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


def train_data(model_path, data):
    script_id = ScriptLangID()
    script_id.train(data)

    # calculate confusion matrix
    result = script_id.identify_many(st for l, st in data)
    result_matrix = build_result_matrix(data, result)

    # using confusion matrix, cluster languages
    cluster = cluster_languages(script_id.lang_vectors, result_matrix, 0.98)

    st_langid_map = {}
    split_data = split_data_by_group(data, cluster)

    for g in cluster:
        if len(g) == 1:
            # don't need to go into furthermore
            continue

        t = split_data[g]

        word_langid = WordLangID(word_entropy_coefficient)
        word_langid.train(iterate_words_st(t))

        st_langid = SentenceLangID(sentence_entropy_coefficient, word_langid)
        st_langid.train(t)

        st_langid_map[g] = st_langid

    save_model(model_path, cluster, script_id, st_langid_map)


def classify_data(model_path, data):
    cluster, script_id, st_langid_map = load_model(model_path)
    split_data = split_data_by_group(data, cluster)
    id_result = {}
    for g in cluster:
        d = split_data[g]
        if len(d) == 0:
            # no sample due to sparse data
            continue

        if len(g) == 1:
            # identification is finished here for those languages
            l = next(iter(g))
            correct = 0
            total = 0
            for l2, st in d:
                if l == l2:
                    correct += 1
                total += 1
            id_result[l] = (correct, total)
        else:
            st_langid = st_langid_map[g]

            result = st_langid.identify_many(st for _, st in d)
            st_result_matrix = build_result_matrix(d, result)
            for l, dist in st_result_matrix.items():
                correct = dist[l]
                total = sum(dist.values())
                id_result[l] = (correct, total)

        script_name = set(script_map.script_str(d[0][1]))
        script_name.discard('Common')
        script_name.discard('Inherited')

        print('------ {} ------'.format(script_name))
        for l in g:
            correct, total = id_result.get(l, (0, 0))
            if total > 0:
                print(l, correct, total, correct/total)
            else:
                print(l, correct, total, 'N/A')
        if len(g) > 1:
            transposed = zip(*(id_result[l] for l in g if l in id_result))
            correct, total = tuple(sum(x) for x in transposed)
            print(script_name, correct, total, correct/total)

        #print_result_matrix(st_result_matrix)

    correct, total = tuple(sum(x) for x in zip(*id_result.values()))
    print('Sum', correct, total, correct/total)

def main():
    parser = argparse.ArgumentParser(
        description='Language identifier.')
    parser.add_argument('--model', dest='model_path',
                        default='model.pkl',
                        help='a model file path')
    parser.add_argument('--mode', dest='mode', choices=['train', 'classify'],
                        help='train or classify')
    parser.add_argument('--word_coeff', dest='wcoeff',
                        type=float, help='word level entropy coefficient')
    parser.add_argument('--st_coeff', dest='stcoeff',
                        type=float, help='sentence level entropy coefficient')
    args = parser.parse_args()

    train, dev, test = load_dataset()
    global word_entropy_coefficient, sentence_entropy_coefficient
    word_entropy_coefficient = args.wcoeff
    sentence_entropy_coefficient = args.stcoeff
    if args.mode == 'train':
        train_data(args.model_path, train)
    else:
        classify_data(args.model_path, test)

if __name__ == '__main__':
    main()