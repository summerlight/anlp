# Layed approach for language identification of multilingual documents

## Team 11, AlphaDogs
| Name                      | ID         | e-mail           | Theory | Coding | Data | Writing |
|---------------------------|------------|------------------|:------:|:------:|:----:|:-------:|
| Minchul Park              | 8210695817 | minchulp@usc.edu |    ○   |    ○   |   ○  |         |
| Meng-Yu Chung             | 9208398418 | mengyuch@usc.edu |        |    ○   |   ○  |         |
| Nithin Chandrashekhar     |            | nithinch@usc.edu |        |    ○   |   ○  |    ○    |
| Samual Krish Ravichandran | 6334599483 | samualkr@usc.edu |    ○   |    ○   |      |    ○    |

## Introduction

Language identification(LID) plays a key part in many NLP tasks that require linguistic assumptions, including machine translation. This task can be difficult; even commercial-level translators, like Google translate, often confuse among those languages closely related by script(Wallon and French, Punjabi and Hindi/Urdu) or lexicon(Hindi language family).

Our approach is to find those similar language groups in a systematic way by using LID and gradually specialize identifiers to each group by taking specific linguistic knowledge into account. This allows us to break down a hard problem of LID with more than 100 languages into small modularized problems and incrementally improve the overall performance with ease.

Meanwhile, the problem grows in a multilingual scenario. Previous works on multilingual identification are done mostly in a document level[1][1][2][2] and there have been only few attempts on identifying multiple languages in a fine-grained way[3][3][4][4]. As our approach requires exact spans of each language, we will treat it as a sequence-labeling problem.

## Method

### Materials

Our task will be based on the [ALTA-2010 Shared Task][5] dataset. This dataset provides multilingual texts for 74 languages by concatenating comparable monolingual documents. In the case of need for more data/annotations, we will use the same method.

### Procedure

We will build a basic identifier from several standard methods for a sequence labeling problem, such as linear-chain CRFs, structured SVM. In both cases, we will use PyStruct. Based on this (potentially character-based) identifier, we will find similar languages in a systematic fashion:

 * Identify languages from language-pair datasets.
 * Build a weighted graph using error rates between two languages as weight values.
 * Partition the graph to determine similar language sets.

Once we partition sets, we can repeatedly try LID with advanced features and find similar languages until narrowing down to one answer. We cannot assume that all languages in the document are in a same group; this is one reason that we model this problem as a sequence labeling problem. 

We expect that many generic features (like characters, words, N-grams, morphological,... etc) could be shared among identifiers, though language specific features can also be introduced. Furthermore, an ideal configuration for given a set of methods/features could be automatically derived via enumerating the state space.

### Evaluation

For evaluation, we will take averages of word-level F1 scores for each language. A baseline method will be simple dictionary lookup based on word counting.

## References cited

[1]: http://lrec-conf.org/proceedings/lrec2006/pdf/459_pdf.pdf "Reconsidering Language Identification for Written Language Resources, LREC, 2006"
[2]: https://aclweb.org/anthology/Q/Q14/Q14-1003.pdf "Automatic Detection and Language Identification of Multilingual Documents, Tran. ACL, 2014"
[3]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.139.6877&rep=rep1&type=pdf#page=14 "A Fine-Grained Model for Language Identification, Proc. SIGIR, 2007"
[4]: http://tangra.si.umich.edu/~radev/papers/language_identification.pdf "Labeling the Languages of Words in Mixed-Language Documents using Weakly Supervised Methods, NAACL HLT, 2015"
[5]: http://aclweb.org/anthology/U/U10/U10-1003.pdf "Multilingual Language Identification: ALTW 2010 Shared Task Dataset, ALTW, 2010"

This document contains 555 words.
