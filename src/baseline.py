import dataset
from collections import defaultdict
from data_gen import segmentation

def load_dataset():
    data = list(dataset.read_dataset('weighted'))

    train = data[:16000]
    dev = data[16000:18000]
    test = data[18000:]

    return train, dev, test

def extract_all_sentences(data):
    for doc in data:
        for l, st in dataset.iter_sentences(doc):
            yield l, st

train, dev, test = load_dataset()

cnt = defaultdict(lambda: defaultdict(int))

for l, st in extract_all_sentences(train):
    for w in segmentation.by_words(st):
        cnt[w][l] += 1

def find_max(dist):
    max_val = max(x[1] for x in dist.items())
    return frozenset(k for k, v in dist.items() if v == max_val)


most_freq = {
    k: find_max(dist) for k, dist in cnt.items()
}

correct = 0
total = 0
correct_word = 0
total_word = 0

for l1, st in extract_all_sentences(dev):
    st_cnt = defaultdict(int)
    for w in segmentation.by_words(st):
        l = most_freq.get(w, '')
        for i in l:
            st_cnt[i] += 1

        if l1 in l:
            correct_word += 1
        total_word += 1
    l2 = find_max(st_cnt)
    if l1 in l2:
        correct += 1
    total += 1

print(correct / total, \
      correct_word / total_word)