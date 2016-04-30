# Layered approach for language identification of multilingual documents

## Team 11, AlphaDogs
| Name                      | ID         | e-mail           | Theory | Coding | Data | Writing |
|---------------------------|------------|------------------|:------:|:------:|:----:|:-------:|
| Minchul Park              | 8210695817 | minchulp@usc.edu |    ○   |    ○   |   ○  |         |
| Meng-Yu Chung             | 9208398418 | mengyuch@usc.edu |        |    ○   |   ○  |         |
| Nithin Chandrashekhar     |            | nithinch@usc.edu |        |    ○   |   ○  |    ○    |
| Samual Krish Ravichandran | 6334599483 | samualkr@usc.edu |    ○   |    ○   |      |    ○    |

## Introduction

A language is more often comprised of features that extend geographically and social space. Language identification(LID) plays a key part in many Language Processing tasks that require linguistic assumptions, one of them being machine translation. The major question that needs to be addressed : "Is it sufficient to determine the unknown language with the set of the possible language". This task can be difficult; even for commercial-level translators like Google translate or Bing Translator which often wrongly identify languages that are closely related by script (some example are Wallon and French, Punjabi and Urdu, Hindi and Kashmiri) or lexicon (Devanagari family contains about 120 languages that use the same script).

Our approach is to find those similar language groups in a systematic way by gradually include specialized identifiers to each language group by taking specific linguistic knowledge into account. This allows us to break down a hard problem of Language Identification with more than 100 languages into small modularized problems and incrementally improve the overall performance with ease. In addition to this, we also introduce the idea of entropy to determine a discriminative power of each word in the text of language identification. Since entropy is a measure of information contained in a probabilistic distribution, we can use it to distinguish words more useful on language identification task and give more weight on them.

On the other hand, multilingual documents pose a major problem during document translation. Previous works on multilingual identification are done mostly on a document level[1][1][2][2] and there have been only a few attempts at identifying multiple languages in a fine-grained way[3][3][4][4]. Typically, sentence is a minimal text segmentation to maintain monolingual assumption; thus sentence-level granularity is sufficient for the purpose of identifying exact span of each languages in multilingual documents. Hence, the problem is formulated as a sentence-level language identification here.

## Method

### Materials

Our dataset is generated from the Wikipedia corpus. First, we downloaded entire text dump files from the [Wikimedia Downloads](http://dump.wikipedia.org) site and removed irrelevant texts, such as tags or URLs, by using [wikiextractor](https://github.com/attardi/wikiextractor). Then we excluded namespaced pages, as many of those pages in minor languages are occasionally written in English. We assumed all documents in the corpus are monolingual; though this is not necessarily true.

In order to embrace lexical ambiguity that arises from similar languages, we collected comparable monolingual documents by using interlanguage links, which member of them deals with the same topic in a different language. We randomly sampled sentences from the documents and concatenated them in a sentence level. Sentence sampling have been done in favor of minor languages, in order to ensure representativeness of each language in the perspective of the corpus size.

The resulting dataset contains 20,000 documents of 123 languages, each of them consists of maximum 3 languages and 10 to 15 sentences. Each sentence contains minimum 20 characters. These documents are separated into three sets: 16,000 for training, 2,000 for development and 2,000 for a test.

### Procedure

Since sentence-level granularity is sufficient for most of the multilingual documents, we formulate this problem as a sentence-level LangID task. [Unicode text segmentation algorithm](http://unicode.org/reports/tr29/) is used for detecting sentence/word-level boundary detection.

The system is composed of several components. The first component is a script-level identifier. This component tries to identify a language by using only Unicode script information. We calculated the count of Unicode script for each document and language, and built corresponding vectors based on this information. Then we calculate similarities between language and document vector via cosine similarity and select the language that yields the highest score.

V_s = <C(s, script_0), ... , C(s, script_n)>
V_l = sum V_si where si in l
result(s) = argmax_l cosine(V_s, V_l)

C(s, script_n) is the number of codepoints in a sentence s that belongs to Unicode script n. V_s and V_l are vector representations of a sentence s and a language l. A language l that maximizes cosine(V_s, V_l) is selected as an identification result of a sentence s.

Since Unicode script itself does not provide sufficient information for language identification, the result is supposed to be highly erroneous. The agglomerative clustering algorithm is used for finding a cluster structure that minimizes the error between clusters. This algorithm  Documents that belongs to the found clusters are subject to the next language identification task.

The next stage is a word-level identification. In this stage, we incorporate a concept of entropy for gauging discriminative power of each word. The idea is that if we have probability distribution of a certain word over languages, its entropy value will display its informational value on language identification task. For instance, a low entropy value of the distribution means that the appearance of this word is focused on a small language set, therefore this word have more weight over the other words.

E(w) = -sum_l_w(p_wl * log(p_wl))
score(w, l) = c_wl / (E(w) + C_e)
score(st, l) = sum_w_st score(w, l)
result(st) = argmax_l score(st, l)

p_wl and c_wl is a probability and an occurrence count of the word w appearing in a sentence written in language l. E(w) is an entropy value of a word w. C_e is an entropy coefficient for the purpose of both avoiding division-by-zero and weighting. A language l that maximizes a summation value of c_wl divided by the entropy value of w is selected as an language estimation of a sentence s. For this task, only unigram model is used.

For the case of unknown words, we also build a word-level language identifier. In order to discover morphological properties of a corresponding language, we collect every possible substring for all words in each language. Word models are constructed as vectors over these morphological statistics. The models are also augmented by entropy values of each substring.

V_w = <C(w, substr_l0), ... , C(w, substr_ln)>
V_l = sum V_si where si in l
p_wl = cosine(V_w, V_l) / sum_l cosine(V_w, V_l)
c_wl = p_wl / |S_l|

substr_l is a set of all possible substrings observed in the language l. V_w and V_l are vector representations of a word w and a morphological model for language l. We estimate probability distribution p_wl from cosine similarity values. Occurrence count values c_wl are normalized in order to fit their average value to 1. For the sake of efficiency, we limit the number of candidate languages to 5 for each word in this research.


### Evaluation

We evaluate our system described using a sentence-level accuracy of the predicted language labels, along with a baseline method of simple dictionary lookup based on word counting.

## Results

| Baseline | Without entropy | Without word | Use all |
|:--------:|:---------------:|:------------:|:-------:|
|   6.12%  |      65.06%     |    89.99%    | 90.65%  |

The baseline system has a low performance of 6.12% as shown in the table x. This is an expected result since the corpus has a considerable amount of lexical ambiguity due to the way it is generated.

Greek : el
Han : zh
Gujarati : gu
Armenian : hy
Oriya : or
Arabic : ar, fa, arz, ur, mzn, ckb, pnb
Kannada : kn
Devanagari : ne, new, sa, mr, hi
Cyrillic : ce, os, mk, be, uk, sah, kk, ba, ky, mn, tg, bg, sr, ru, cv, tt
Hiragana, Katakana, Han : ja
Myanmar: my
Gurmukhi: pa
Sinhala : si
Malayalam : ml
Hangul : ko
Georgian : ka
Telugu : te
Hebrew : yi, he
Bengali : bn, bpy
Latin : vo
Tamil : ta
Latin : pl, scn, li, sw, it, nah, wa, ast, lv, ceb, fy, en, ht, sq, sv, mg, yo, fo, lmo, ms, nn, war, oc, sl, ca, de, bs, br, cs, af, ku, ro, la, lb, fi, io, eo, jv, sh, nds, pt, et, hsb, eu, hr, hu, vi, es, is, gd, az, ia, da, vec, ga, nl, tr, tl, lt, uz, gl, sk, qu, nap, min, pms, su, an, cy, sco, id, bar, fr,
Thai : th
Ethiopic : am

The script-level identifier serves its purpose almost perfectly. The table x shows the generated clusters based on the result. In consideration of this identifier's original purpose is to divide the dataset according to their scripts, we can see that this objective is successfully achieved. However, further attempts to recursively cluster languages using same script was not rewarding. It is discussed in the next section in detail.

The accuracies of word-level language identifiers for each script are described in the table x. While it is extremely challenging to estimate the language solely from a word itself without furthermore context, our result indicates that rather inaccurate estimation can bring performance improvement for sentence-level language identification task. Although total improvement is marginal because we does not give a huge weight on unknown words, the performance gain on Arabic and Cyrillic languages is noticeable.

| word \ sentence |  0.1   |  0.01  |  0.001 |
|:---------------:|:------:|:------:|:------:|
|       0.1       | 80.32% | 87.69% | 90.65% |
|      0.01       | 80.27% | 87.64% | 90.60% |
|     0.001       | 80.23% | 87.61% | 90.59% |

We measured the performance of a sentence-level identifier with various entropy coefficient settings. As described in the table x, the result shows effectiveness of our approach. Using entropy of each word to give weight gives substantial performance gain of 15% and further parameter tuning yields 10% of improvement. While 90% of accuracy is not a very satisfactory result for a language identification task, it is worth to note that we only used unigram feature. Since we initially focused on enabling gradual improvement, we believe this result can be easily improved if the clustering problem is solved.


## Discussion

Our core contributions in this research can be summarized in two parts. The first one is emphasizing a discriminative power of certain features by using entropy of corresponding probability distribution over languages. The second part is dividing a larger problem into smaller problems by clustering similar languages.

The first part works fairly well. The primary reason is that our dataset is generated from similar documents so simple . For instance, the word "York" has a very high entropy value, because use of "New York" is so prevalent regardless of languages. In our observation, named entities usually have low discriminative power whereas long adjective/adverbs are opposite. We think future studies of language identification utilizing full dictionary/tag information may yield these kinds of interesting information. Also, investigation of this idea on other classification tasks could be fruitful.

Although clustering for finding similar languages worked almost perfectly on the script level, we realized that a more sophisticated clustering scheme is required for the later stage. While we could find several obvious clusters such as a Serbo-Croatian family (sh, hsb, bs) in the Latin script cluster, our relatively simple algorithm was not able to appreciate them. Due to data sparseness of minor languages, their characteristics are insufficiently represented. Thus, they are occasionally misidentified in a large set of random languages. The algorithm that we used for the task is greedy, so it tries to merge two clusters that can eliminate as many errors as possible in a single step. When the algorithm eventually merges the minor language, then merging based on error rate propagates to all the other languages. In the result, we have a single large set of irrelevant languages. Future works may deal with this problem.

On the other side, we found the surprising result; our relatively simple model using cosine similarity outperformed other sophisticated models by a noticeable margin. While we were not able to present concrete numbers due to the schedule constraint, we initially experimented with some other models including HMM, MEMM and CRF. Our conclusion is that they may not work very well without appropriate understanding of the model, precise tuning of hyperparameters and thoughtful feature engineering.

Also, it is worth to note that many advanced models demand significant computational power, especially in this task mainly due to the huge number of vocabularies. For instance, nltk Naive-Bayes classifier crashed on the full dataset due to OOM error even with 16gb of memory. A five-gram model on CRF generated billions of features which most CRF implementations are unable to deal with and even a three-gram model took 20~30 minutes to train the entire dataset. In the perspective of a problem size, we expect solving the clustering problem stated above can alleviate this computational problem.

## References cited

[1]: http://lrec-conf.org/proceedings/lrec2006/pdf/459_pdf.pdf "Reconsidering Language Identification for Written Language Resources, LREC, 2006"
[2]: https://aclweb.org/anthology/Q/Q14/Q14-1003.pdf "Automatic Detection and Language Identification of Multilingual Documents, Tran. ACL, 2014"
[3]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.139.6877&rep=rep1&type=pdf#page=14 "A Fine-Grained Model for Language Identification, Proc. SIGIR, 2007"
[4]: http://tangra.si.umich.edu/~radev/papers/language_identification.pdf "Labeling the Languages of Words in Mixed-Language Documents using Weakly Supervised Methods, NAACL HLT, 2015"
[5]: http://aclweb.org/anthology/U/U10/U10-1003.pdf "Multilingual Language Identification: ALTW 2010 Shared Task Dataset, ALTW, 2010"

This document contains xxx words.
