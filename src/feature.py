import unicodedata
import itertools
import script_map
import glob
import json
from data_gen import segmentation
from data_gen import altw_adaptor


# The purpose of this module is generation of word/character-wise features.
def s(ch):
    return script_map.script(ch)


def p(ch):
    return hex(ord(ch))


def c(ch):
    return unicodedata.category(ch)


def tagged_seq(doc):
    metadata = doc['metadata']
    text = doc['text']
    for segment in metadata:
        # TODO: should we consider chars that do not belong to any segment?
        for i in range(segment['begin'], segment['end']):
            yield segment['lang'], text[i]


def read_dataset():
    for file_path in glob.glob('../data/sentences/*.txt'):
        with open(file_path, 'rt') as f:
            yield json.load(f)


def iter_doc(doc):
    text = doc['text']
    for segment in doc['metadata']:
        yield segment['lang'], text[segment['begin']:segment['end']]


def is_common(word):
    return all(i == 'Common' for i in script_map.script_str(word))


# escaping is needed because wapiti parses some characters wrongly
def escape(word):
    return ''.join(hex(ord(i)) for i in word)


def tag_document(doc):
    for lang, text in iter_doc(doc):
        for word in segmentation.by_words(text):
            if word.isspace():
                continue
            if is_common(word):
                yield 'zz', word
            else:
                yield lang, word


with open('input.txt', 'wt') as f:
    for doc in itertools.islice(read_dataset(), 1000):
        for lang, word in tag_document(doc):
            f.write('{}\t{}\n'.format(escape(word), lang))
        f.write('\n')
