import sqlite3
import argparse
import random
import json
import os.path
import itertools
from collections import defaultdict

'''
- Multilingual document format

Every multilingual document is a sequence of language-annotated text segments
as below:

multi_doc = [
    {'lang':'en',
     'text':'Hello world!'},
    {'lang':'fr',
     'text':'Bonjour monde!'},
    ...
]

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

# This script does not generate actual multilingual text files, but provides
# some functionalities for generate dataset. This is because generating dataset
# itself takes lots of running time and lots of space while we haven't fixed
# our strategy for data generation. Until we have a fixed dataset, data
# generation on the fly is good enough for the purpose.


# Utility functions
# TODO: maybe those can be moved into other module?
def raw_text(annotated_doc):
    return ''.join(segment['text'] for segment in annotated_doc)


def write_json(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wt') as f:
        json.dump(obj, f)


def read_text(doc_info):
    path = doc_info['doc_path']
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
def combine_text(doc_tuple):
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
    p1 = p1[len(p1) // 2:]
    p2 = p2[:len(p2) // 2]

    return [
        {'lang': l1, 'text': '\n'.join(p1)},
        {'lang': l2, 'text': '\n'.join(p2)},
    ]


def generate_documents(doc_infos, lang_selector, doc_generator):
    texts = {
        doc_info['lang']: read_text(doc_info) for doc_info in doc_infos
    }
    langs = [doc_info['lang'] for doc_info in doc_infos]

    for lang_tuple in lang_selector(langs):
        yield doc_generator(tuple((l, texts[l]) for l in lang_tuple))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Locate same topic texts over multiple languages.')
    parser.add_argument('--input', dest='input_path',
                        default='interlanguage_location.json',
                        help='a json file generated by locate_text.py')
    parser.add_argument('--output', dest='output_path',
                        default='./output/',
                        help='an output path for generated documents')
    args = parser.parse_args()

    with open(args.input_path, 'rt') as f:
        location_infos = json.load(f)

    for doc_infos in location_infos:
        # TODO: Actually, we don't want to generate all possible combinations
        #       but want to reflect actual distribution as ALTA-2010
        for annotated in generate_documents(doc_infos,
                                            iter_all_pairs,
                                            combine_half_by_half):
            print(annotated)
