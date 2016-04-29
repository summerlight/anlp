import sqlite3
import argparse
import random
import json
import os.path
import itertools
import bisect
import segmentation
from collections import defaultdict, Sequence

'''
- Multilingual document format

Every multilingual document is a sequence of language-annotated text segments
as below:

multi_doc = {
    'text': 'Hello world! Bonjour monde!'
    'metadata': [
        {'lang':'en',
         'begin': 0,
         'end': 13},
        {'lang':'fr',
         'begin': 13,
         'end': 26},
        ...
    ]
}

which is 'Hello world!Bonjour monde!' in a raw text format. Note that we don't
add any artificial padding between each segment. If we need it, then it should
be handled explicitly.

The primary reason for this decision that it is simple to treat this structure
as a JSON format. Although we could use Python pickle for the purpose, we want
to avoid some potential portability issues on using it as our serialization
format.

* TODO : generation strategy
 - paragraph level
 - sentence level
 - random mixture
 - more than 2 languages?
 - consideration of context
 - etc...
'''

# a languages set to filter out languages without enough articles
lang_candidates = {
'de', 'en', 'es', 'fr', 'it', 'nl', 'ja', 'pl', 'ru', 'ceb', 'sv', 'vi', 'war',
# more than 100,000 pages
'ar', 'az', 'bg', 'zh-min-nan', 'be', 'ca', 'cs', 'da', 'et', 'el', 'eo', 'eu',
'fa', 'gl', 'ko', 'hy', 'hi', 'hr', 'id', 'he', 'ka', 'la', 'lt', 'hu', 'ms',
'min', 'no', 'nn', 'ce', 'uz', 'pt', 'kk', 'ro', 'sk', 'sl', 'sr', 'sh', 'fi',
'th', 'tr', 'uk', 'ur', 'vo', 'zh',
# more than 10,000 pages
'af', 'als', 'am', 'an', 'ast', 'bn', 'map-bms', 'ba', 'be-tarask', 'bpy',
'bar', 'bs', 'br', 'cv', 'cy', 'fo', 'fy', 'ga', 'gd', 'gu', 'hsb', 'io', 'ia',
'os', 'is', 'jv', 'kn', 'ht', 'ku', 'ckb', 'ky', 'mrj', 'lv', 'lb', 'li',
'lmo', 'mk', 'mg', 'ml', 'mr', 'arz', 'mzn', 'mn', 'my', 'nah', 'new', 'ne',
'nap', 'oc', 'or', 'pa', 'pnb', 'pms', 'nds', 'qu', 'sa', 'sah', 'sco', 'sq',
'scn', 'si', 'su', 'sw', 'tl', 'ta', 'tt', 'te', 'tg', 'bug', 'vec', 'wa',
'yi', 'yo', 'zh-yue', 'bat-smg'}

# This script does not generate actual multilingual text files, but provides
# some functionalities for generate dataset. This is because generating dataset
# itself takes lots of running time and lots of space while we haven't fixed
# our strategy for data generation. Until we have a fixed dataset, data
# generation on the fly is good enough for the purpose.

# TODO: Actually, we don't want to generate all possible combinations
#       but want to reflect actual distribution as ALTA-2010


# Utility functions
# TODO: maybe those can be moved into other module?
def write_json(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wt') as f:
        json.dump(obj, f)


def read_text(doc_info, path):
    path = os.path.join(path, doc_info['doc_path'])
    begin = doc_info['begin']
    end = doc_info['end']
    with open(path, 'rt') as f:
        return f.read()[begin:end]


# iterate over all possible pairs
def iter_all_pairs(langs):
    for l1, l2 in itertools.product(langs, langs):
        if l1 != l2:
            yield l1, l2


# combine two texts by following ALTA-2010's methodology
def combine_half_by_half(doc_tuple):
    # we use only first of two texts here
    assert len(doc_tuple) >= 2
    l1 = doc_tuple[0][0]
    t1 = doc_tuple[0][1]
    l2 = doc_tuple[1][0]
    t2 = doc_tuple[1][1]

    # the first two lines can be ignored because they're just about title.
    p1 = [i for i in t1.split('\n')[2:] if len(i) > 2]
    p2 = [i for i in t2.split('\n')[2:] if len(i) > 2]

    # split a document by half and concatnate
    # the first half of t1 and the second half of t2.
    p1 = '\n'.join(p1[len(p1) // 2:])
    p2 = '\n'.join(p2[:len(p2) // 2])

    return {
        'text': p1 + '\n' + p2,
        'metadata': [
            {'lang': l1, 'begin': 0, 'end': len(p1)},
            {'lang': l2, 'begin': len(p1) + 1, 'end': len(p1) + len(p2) + 1},
        ]
    }


# sampling
def random_combinations(cnt, lang_cnt):
    def generator(langs):
        seen = set()
        for _ in range(cnt):
            while True:
                sampled = tuple(random.sample(langs, lang_cnt))
                if sampled not in seen:
                    seen.add(sampled)
                    yield sampled
                    break
    return generator


# weight over minor languages
def weighted_combinations(cnt, lang_cnt):
    lang_cnt_dict = defaultdict(int)

    class InvertedPopulation(Sequence):
        def __init__(self, lang):
            total = sum(lang_cnt_dict[i] + 1 for i in lang)
            #avg = total / len(lang)
            weights = (int(total / (lang_cnt_dict[i] + 1)) for i in lang)

            self.pop = list(lang)
            self.weights = list(itertools.accumulate(weights))

        def __len__(self):
            return self.weights[-1]

        def __getitem__(self, i):
            if not 0 <= i < len(self):
                raise IndexError(i)
            return self.pop[bisect.bisect(self.weights, i)]

    def generator(langs):
        seen = set()
        weighted_pop = InvertedPopulation(langs)

        for _ in range(cnt):
            while True:
                sampled = tuple(random.sample(weighted_pop, lang_cnt))
                if sampled in seen or len(set(sampled)) != lang_cnt:
                    continue
                seen.add(sampled)
                for l in sampled:
                    lang_cnt_dict[l] += 1
                yield sampled
                break
        print(lang_cnt_dict)

    return generator


def combine_sentences(doc_tuple):
    text = ''
    metadata = []
    for _ in range(random.randrange(10, 15)):
        doc = random.choice(doc_tuple)
        sentences = [st for st in segmentation.by_sentences(doc[1])
                     if len(st) > 10]
        picked = random.choice(sentences)
        begin = len(text)
        text += picked
        metadata.append({
            'lang': doc[0],
            'begin': begin,
            'end': len(text)
        })
    return {
        'text': text,
        'metadata': metadata
    }


def generate_documents(location_infos, path, lang_selector, combiner):
    while True:
        doc_infos = random.choice(location_infos)
        texts = {
            doc['lang']: read_text(doc, path) for doc in doc_infos
        }
        langs = [doc['lang'] for doc in doc_infos]

        for lang_tuple in lang_selector(langs):
            yield combiner(tuple((l, texts[l]) for l in lang_tuple))


def refine_candidates(location_infos):
    def refine_topic(doc_infos):
        return [doc for doc in doc_infos if doc['lang'] in lang_candidates]

    location_infos = [refine_topic(doc_infos) for doc_infos in location_infos]

    # some topics are not indexed because of several reasons
    return [l for l in location_infos if len(l) >= 20]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Locate same topic texts over multiple languages.')
    parser.add_argument('--input', dest='input_path',
                        default='interlanguage_location.json',
                        help='a json file generated by locate_text.py')
    parser.add_argument('--output', dest='output_path',
                        default='../../data/sentences/',
                        help='an output path for generated documents')
    parser.add_argument('--doc_count', dest='doc_count', type=int,
                        default=10000, help='Number of generated documents')
    args = parser.parse_args()
    input_dir = os.path.dirname(args.input_path)

    with open(args.input_path, 'rt') as f:
        location_infos = json.load(f)

    # some topics are not indexed because of several reasons
    location_infos = refine_candidates(location_infos)
    generator = generate_documents(location_infos,
                                   input_dir,
                                   weighted_combinations(10, 3),
                                   combine_sentences)

    for idx, doc in itertools.islice(enumerate(generator), args.doc_count):
        write_json(doc, os.path.join(args.output_path,
                                     'doc{}.txt'.format(idx)))
