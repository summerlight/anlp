import glob
import json
from data_gen import segmentation


def iter_segment(doc):
    text = doc['text']
    for segment in doc['metadata']:
        yield segment['lang'], text[segment['begin']:segment['end']]


def iter_sentences(doc):
    for lang, segment in iter_segment(doc):
        for st in segmentation.by_sentences(segment):
            yield lang, st


def iter_words(doc):
    for lang, segment in iter_segment(doc):
        for word in segmentation.by_words(segment):
            yield lang, word


def iter_words_from_st(st):
    for lang, segment in iter_segment(doc):
        for word in segmentation.by_words(st):
            yield lang, word


def iter_words_by_st(doc):
    for lang, segment in iter_segment(doc):
        for st in segmentation.by_sentences(segment):
            yield lang, (word for word in segmentation.by_words(st))


def iter_chars(doc):
    for lang, segment in iter_segment(doc):
        for ch in segment:
            yield lang, ch


def tagged_words(doc):
    for lang, word in iter_words(doc):
        if word.isspace():
            continue
        if is_common(word):
            yield 'zz', word
        else:
            yield lang, word


def read_dataset(name):
    for file_path in sorted(glob.glob('../data/{}/*.txt'.format(name))):
        with open(file_path, 'rt') as f:
            val = json.load(f)
            val['file_path'] = file_path
            yield val