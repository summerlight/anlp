import unicodedata
import itertools
import script_map
import glob
import json
import pprint
import datset
from collections import defaultdict
from data_gen import segmentation


lang_dict = defaultdict(lambda: defaultdict(int))
word_dict = defaultdict(lambda: defaultdict(int))
prefix_dict = defaultdict(lambda: defaultdict(int))
suffix_dict = defaultdict(lambda: defaultdict(int))
lang_script_dict = defaultdict(lambda: defaultdict(int))
script_lang_dict = defaultdict(set)


# The purpose of this module is generation of word/character-wise features.
def s(ch):
    return script_map.script(ch)


def p(ch):
    return hex(ord(ch))


def c(ch):
    return unicodedata.category(ch)


def is_common(word):
    return all(i == 'Common' for i in script_map.script_str(word))


def max_key(dic):
    return max(dic.items(), key=lambda x: x[1])[0]


def add_dict(target, source):
    for k, v in source.items():
        target[k] += v


# escaping is needed because wapiti parses some characters wrongly
def escape(word):
    return ''.join(hex(ord(i)) for i in word)


# the first character's script should be fine for the purpose...
def script(word):
    #return next(script_map.script_str(word))
    return '|'.join(set(script_map.script_str(word)))


def likely(word):
    lower = word.lower()
    if lower in word_dict:
        return max_key(word_dict[lower])
    else:
        return 'zz'


def prefixes(word, cnt):
    for i in range(min(len(word), cnt)):
        yield word[:i+1]


def suffixes(word, cnt):
    for i in range(min(len(word), cnt)):
        yield word[-i:]


def write_crfpp_data(file_name, docs):
    with open(file_name, 'wt') as f:
        for doc in docs:
            for lang, word in dataset.tagged_words(doc):
                if lang != 'zz':
                    scr = script(word)
                    f.write('{}\t{}\t{}\n'.format(word, scr, lang))
            f.write('\n')


data = list(dataset.read_dataset('132_langs'))
train = data[:8000]
dev = data[8000:9000]
test = data[9000:]

for doc in train:
    for lang, word in dataset.tagged_words(doc):
        scr = script(word)
        word = word.lower()
        lang_script_dict[lang][scr] += 1
        lang_dict[lang][word] += 1
        word_dict[word][lang] += 1
        for prefix in prefixes(word, 5):
            prefix_dict[prefix][lang] += 1
        for suffix in suffixes(word, 5):
            suffix_dict[suffix][lang] += 1

most_likely = {
    word: max_key(langs) for word, langs in word_dict.items()
}

def estimate(word):
    prefix_cnt = defaultdict(int)
    for prefix in prefixes(word, 5):
        if prefix not in prefix_dict:
            continue
        add_dict(prefix_cnt, prefix_dict[prefix])
    print(prefix_cnt)

    suffix_cnt = defaultdict(int)
    for suffix in suffixes(word, 5):
        if suffix not in suffix_dict:
            continue
        add_dict(suffix_cnt, suffix_dict[suffix])
    print(suffix_cnt)

#pprint.pprint(lang_script_dict)


for lang, scripts in lang_script_dict.items():
    tot_cnt = sum(scripts.values())
    script_set = set()
    for scr, cnt in scripts.items():
        threshold = 0.03 if scr == 'Latin' else 0.01
        if cnt / tot_cnt > threshold:
            script_set.add((scr, cnt / tot_cnt))
    #print(lang, script_set)

write_crfpp_data('train.txt', train)
write_crfpp_data('dev.txt', dev)
write_crfpp_data('test.txt', test)

#print(most_likely)
#print(suffix_dict['ed'])
#estimate('surplused')