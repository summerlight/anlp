# Layered approach for language identification of multilingual documents

## Team 11, AlphaDogs
| Name                      | ID         | e-mail           | Theory | Coding | Data | Writing |
|---------------------------|------------|------------------|:------:|:------:|:----:|:-------:|
| Minchul Park              | 8210695817 | minchulp@usc.edu |    ○   |    ○   |   ○  |         |
| Meng-Yu Chung             | 9208398418 | mengyuch@usc.edu |        |    ○   |   ○  |         |
| Nithin Chandrashekhar     |            | nithinch@usc.edu |        |    ○   |   ○  |    ○    |
| Samual Krish Ravichandran | 6334599483 | samualkr@usc.edu |    ○   |    ○   |      |    ○    |

## Introduction

A language is more often comprised of features that extend geographically and social space. Language identification(LID) plays a key part in many Language Processing tasks that requires linguistic assumptions, one of them being machine translation. The major question that needs to be addressed : "Is it sufficient to determine the unknown language with the set of the possible language". This task can be difficult; even for commercial-level translators like Google translate or Bing Translator which often wrongly identify languages that are closely related by script (some example are Wallon and French, Punjabi and Urdu, Hindi and Kashmiri) or lexicon (Devanagari family contains about 120 language that use the same script).

Our approach is to find those similar language groups in a systematic way by gradually include specialized identifiers to each language group by taking specific linguistic knowledge into account. This allows us to break down a hard problem of Language Identification with more than 100 languages into small modularized problems and incrementally improve the overall performance with ease. This reduces the problem into different identification types based on difficulty level (closely related language - [Dialects with a region] and easily identifiable language - [English]).

On the other hand, multilingual documents pose a major problem during document translation. Previous works on multilingual identification are done mostly on a document level[1][1][2][2] and there have been only a few attempts at identifying multiple languages in a fine-grained way[3][3][4][4]. In the approach that is described in this report, the problem is addressed using exact sentence spans of each language, which can be treated as a sequence-labeling problem.

## Method

### Materials

Our task is based on the [ALTA-2010 Shared Task][5] dataset. This dataset provides multilingual texts for 74 languages by concatenating comparable monolingual documents. In the case of need for more data/annotations, we will use the same method.

### Procedure

We will build a basic identifier from several standard methods for a sequence labeling problem, such as linear-chain CRFs, structured SVM. In both cases, we will use PyStruct. Based on this (potentially character-based) identifier, we will find similar languages in a systematic fashion:

 * Identify languages from language-pair datasets.
 * Build a weighted graph using error rates between two languages as weight values.
 * Partition the graph to determine similar language sets.

Once similar language sets are achieved, we can repeatedly try LID with advanced features and categorize similar languages until narrowing down to one answer. We cannot assume that all languages in a document are in the same group; it is one reason of this being a sequence labeling problem.

We expect that many generic features (like characters, words, N-grams, etc.) can be shared among identifiers while language specific features can also be introduced. Furthermore, an ideal configuration for a given set of features could be automatically derived by enumerating possible state space.

### Evaluation

For evaluation, we will take averages of word-level F1 scores for each language. A baseline method will be simple dictionary lookup based on word counting.

## Results

Report how your system performs, and how it compares to the baseline or to other comparable work. Discuss what it gets right, what it gets wrong, and why.

## Discussion

Discuss conclusions that can be drawn from the research, implications of your findings, the overall contribution to the general NLP community, and directions for future research.

## References cited

[1]: http://lrec-conf.org/proceedings/lrec2006/pdf/459_pdf.pdf "Reconsidering Language Identification for Written Language Resources, LREC, 2006"
[2]: https://aclweb.org/anthology/Q/Q14/Q14-1003.pdf "Automatic Detection and Language Identification of Multilingual Documents, Tran. ACL, 2014"
[3]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.139.6877&rep=rep1&type=pdf#page=14 "A Fine-Grained Model for Language Identification, Proc. SIGIR, 2007"
[4]: http://tangra.si.umich.edu/~radev/papers/language_identification.pdf "Labeling the Languages of Words in Mixed-Language Documents using Weakly Supervised Methods, NAACL HLT, 2015"
[5]: http://aclweb.org/anthology/U/U10/U10-1003.pdf "Multilingual Language Identification: ALTW 2010 Shared Task Dataset, ALTW, 2010"

This document contains xxx words.
