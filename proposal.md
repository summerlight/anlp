# Language identification of multi-lingual documents

## Team 11, AlphaDogs
| Name                      | USC ID     | USC e-mail       |
|---------------------------|------------|------------------|
| Minchul Park              | 8210695817 | minchulp@usc.edu |
| Meng-Yu Chung             | 9208398418 | mengyuch@usc.edu |
| Nithin Chandrashekhar     |            | nithinch@usc.edu |
| Samual Krish Ravichandran | 6334599483 | samualkr@usc.edu |

## Introduction

While language identification(LID) is a well-studied problem, LID of multilingual documents still remains as an open problem. [Linguini][1] is the first known work that tackles this topic by giving proportion of languages. [Hughes et al. (2006)][2] proposed several challenges for LID, including multilingual documents. While many of previous works on multilingual LID are done in a document level[1][1][3][3][4][4], there have been several attempts on identifying multiple languages in a fine-grained way[5][5][6][6].

Here, our problem formulation is a fine-grained, word-level sequence labeling (and potentially identifying language segmentations). While this is a natural problem formulation, identifying exact spans of monolingual text can simplify many NLP tasks. For instance, indexing vocabularies of web documents usually depends on language-specific morphological transformation like stemming, therefore identifying a corresponding language of an indexed word is essential for this task.

## Method

### Materials

Since there are no standard corpora for this task, we will use Wikipedia corpus to construct multilingual documents from monolingual sentences. This corpus contains about 200gb of text data, more than 100 languages, which should be sufficient for our purpose.

To be specific, several real-world scenarios are in our consideration for dataset construction:

 * By structure - concatenation of short sentences, foreign words in a sentence, foreign sentences in a paragraph
 * By language - choosing randomly, choosing among closely related languages
 * By contents - sentences from same topic but different languages (Wikipedia provides this data)

Currently, we plan to enumerate every possible configuration for data generation. 

### Procedure

As there are not many works done before for this problem, we will try several standard methods for a sequence labeling problem, such as [linear-chain CRFs][7], [structured SVM][8] and/or [LSTM RNN][9]. In this case, we think character, word, N-grams would be all valid features.

Python has a plentiful number of implementation of those machine learning algorithms. [PyStruct][10] supports CRFs and structured SVM and [CRF++][11] provides a framework for CRFs. [Theano][12] is a powerful tool for implementing RNN by exploiting GPGPU power.

There can be two possible consequences. If the basic implementation is good enough and our schedule allows, then we will tackle other challenges including those proposed by [Hughes et al. (2006)][2]:

 * Open class LID, or identifying unknown language
 * Training from sparse datasets
 * Include more minor languages
 * Unsupervised training
 * Experimenting our system on language dialects or code switching datasets
 
Otherwise, we will focus on improving our system by better feature engineering, finding better modeling, ensemble methods, etc. One idea is layered classification. The motivation here is that there should be sets of closely related, or confusing languages, and we can identify such sets by (possibly monoligual) LID. Hence, we identify spans of a potential language set first, then identify actual language spans with more sophisticated, language-specific features. This approach is natural since it can reflect usual abstraction layers for text representation like character set, vocabulary, morphology. 

### Evaluation

For evaluation, we will measure word-level F1 scores. A baseline method will be a simple dictionary lookup based on word counting.

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

## Division of labor between the teammates

| Name    | Theory | Coding | Data | Writing |
|---------|:------:|:------:|:----:|:-------:|
| Minchul |    ○   |    ○   |   ○  |         |
| Meng-Yu |        |    ○   |   ○  |         |
| Nithin  |        |    ○   |   ○  |    ○    |
| Samual  |    ○   |    ○   |      |    ○    |

This document contains xx words.
