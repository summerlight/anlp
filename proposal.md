# Layed approach for language identification of multilingual documents

## Team 11, AlphaDogs
| Name                      | USC ID     | USC e-mail       | Theory | Coding | Data | Writing |
|---------------------------|------------|------------------|:------:|:------:|:----:|:-------:|
| Minchul Park              | 8210695817 | minchulp@usc.edu |    ○   |    ○   |   ○  |         |
| Meng-Yu Chung             | 9208398418 | mengyuch@usc.edu |        |    ○   |   ○  |         |
| Nithin Chandrashekhar     |            | nithinch@usc.edu |        |    ○   |   ○  |    ○    |
| Samual Krish Ravichandran | 6334599483 | samualkr@usc.edu |    ○   |    ○   |      |    ○    |

## Introduction

Machine Translation Engines needs to the identify the source language before it could be translated into target language. Language identification(LID) plays a key part in the machine translation so that accurate translation can be made. While LID is a well-studied problem, its application to multilingual documents still remains as an open problem. The problem that we are trying to address are with languages which have same scripting like for example, Wallon Language will be identified as French (in Google Translate) since it is French language with the same script but different grammar and vocabulary, or different root like for example, Punjabi Language has two root from Hindi and Urdu scripts. The problem at hand is to take into account the similarity between language and its descended scripts which uses the same construct and estimate in terms of vocabulary and sentence construction which could give much meaning sentences and thus helps in determining the language to be identified. Even though many of previous works on multilingual LID are done in a document level[1][1] [3][3] [4][4], there have been several attempts on identifying multiple languages in a fine-grained way[5][5] [6][6] inside a single document. The motivation is that identifying exact spans of monolingual texts can simplify many NLP tasks. The problem was first addressed by [Linguini][1] by giving proportion of languages. Consequently [Hughes et al. (2006)][2] proposed several challenges for LID, including multilingual documents. 

<del>TODO : how to integrate below paragraphs?</del>
By this means, we're expecting below benefits:

 * Breaking down a hard problem (LID with 100~ languages) into smaller, modularized problems.
 * We can easily experiment very specific features without a fear of degrading entire system's performance.
 * This approach naturally maps to usual abstractions for textual representations.
 * This approach allows us to replace a LID for specific languages, if drop-in replacement with better performance exists.

<del>While the below paragraph is too long, we're still required to put some references, and that our problem is a sequence labeling problem.</del>
While LID is a well-studied problem, its application to multilingual documents still remains as an open problem. [Linguini][1] is the first known work that tackles this topic by giving proportion of languages. [Hughes et al. (2006)][2] proposed several challenges for LID, including multilingual documents. While many of previous works on multilingual LID are done in a document level[1][1][3][3][4][4], there have been several attempts on identifying multiple languages in a fine-grained way[5][5][6][6]. For LID itself, our problem formulation is a fine-grained, word-level sequence labeling (and potentially identifying language segmentations) for multilingual documents. The motivation is that identifying exact spans of monolingual texts can simplify many NLP tasks. 

## Method

### Materials

Our task will be based on the [ALTA-2010 Shared Task][ALTA-2010] dataset. This dataset provides multilingual texts for 74 languages by concatenating comparable monolingual documents. In the case of need for more data/annotations, we will follow the same methodology.

### Procedure

We will build a basic identifier from several standard methods for a sequence labeling problem, such as [linear-chain CRFs][7], [structured SVM][8]. In both cases, we will use [PyStruct][10]. Based on this (potentially character-based) identifier, we will find similar languages in a systematic fashion:

 * Identify languages from language-pair datasets.
 * Build a weighted graph using error rates between two languages as weight values.
 * Partition the graph to determine similar language sets.

Once we have partitioned sets, we can repeatedly try LID with advanced features and find similar languages until narrowing down to one answer. We need exact spans of each language for this task; this is one reason that we model this problem as a sequence labeling problem. 

We expect that many generic features (like characters, words, N-grams, morphological,... etc) could be shared among identifiers, though more language specific features can also be introduced. Furthermore, an ideal configuration for given a set of methods/features could be automatically derived via enumerating the state space.

### Evaluation

For evaluation, we will measure averages of word-level F1 scores for each language. A baseline method will be simple dictionary lookup based on word counting.

## References cited

[1]: http://www.computer.org/csdl/proceedings/hicss/1999/0001/02/00012035.pdf "Linguini: Language Identification for Multilingual Documents"
[2]: http://lrec-conf.org/proceedings/lrec2006/pdf/459_pdf.pdf "Reconsidering Language Identification for Written Language Resources"
[3]: https://aclweb.org/anthology/Q/Q14/Q14-1003.pdf "Automatic Detection and Language Identification of Multilingual Documents"
[4]: https://www.researchgate.net/profile/Zakaria_Elberrichi/publication/220531464_Automatic_Language_Identification_An_Alternative_Unsupervised_Approach_Using_a_New_Hybrid_Algorithm/links/0fcfd50cb7bd3ceeef000000.pdf "Automatic language identification: An alternative unsupervised approach using a new hybrid algorithm"
[5]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.139.6877&rep=rep1&type=pdf#page=14 "A Fine-Grained Model for Language Identification"
[6]: http://tangra.si.umich.edu/~radev/papers/language_identification.pdf "Labeling the Languages of Words in Mixed-Language Documents using Weakly Supervised Methods"
[7]: https://www.cs.utah.edu/~piyush/teaching/crf.pdf "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"
[8]: http://www.jmlr.org/papers/volume6/tsochantaridis05a/tsochantaridis05a.pdf "Large Margin Methods for Structured and Interdependent Output Variables"
[9]: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf "LONG SHORT-TERM MEMORY"
[10]: https://pystruct.github.io/ "PyStruct - Structured Learning in Python"
[11]: https://taku910.github.io/crfpp/ "CRF++: Yet Another CRF toolkit"
[12]: https://github.com/Theano/Theano "Theano"
[ALTA2010]: http://aclweb.org/anthology/U/U10/U10-1003.pdf "ALTA-2010 Shared Task"

This document contains xx words.
