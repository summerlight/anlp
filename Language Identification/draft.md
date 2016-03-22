## Motivation 
  
Usage of interntet is growing at an exponential rate and so does the number of users. These users are from different parts of the world and speak different languages. The content of a page on a website written by an user from one part of the world will not be understood by an user of the website from a different part of the world.

The main problem of not being able to understand the web page is due to different languages being used. If an author writes a web page about a topic in his native language and if users from other parts of the world need to understand it, they should first be able to identify the language in which the web page is written.

## Problem definition

The problem is basically a classification problem, which we are targeting involves identifying a language based on the corpus we get. We need to take into consideration, the usage of words or sentence and the way these words or sentences are related to each other from the training corpus.

It is known that a performance of language identification can be degraded on some conditions:

1. More languages
1. Less dataset
1. Shorter text
1. Independence of character encodings

In order to make the problem interesting, we will try to do language identification with these constraint. Details will be discussed over the next sections.

Define different classes (languages) based on the training corpus.

## Method
We will collect the training corpus of different languages. We consider each language as a class. Apply Naive Bayes to learn the model based on the training corpus. Use this learned model to classify the development corpus. This will be a baseline for our project.

One of our expectation is that there will be groups of commonly confusing languages; for instance, many of latin-based languages share a common character set. If some document is wrongly classified, the result is likely to be another language in the same group. Therefore we will try to classify documents in a confusing group in a adaptive manner; if the result is in a group of confusing langauges then we will try to classify it with different methods/features engineered specific to that language group. This could be thought of a kind of ensemble learning.

We will use various classification methods like Naive bayes, SVM, K-NN... with various features. Also, we want the whole framework to be as generic and flexible as possible. So if time allows, the framework should automatically try each method/feature and select the best configuration, like Brill's tagger. One of the reason for this decision is that methods that work well could differ for each language, and it takes huge amount of time to experiment each configuration. We will use unigram as a basic feature and progressively deploy more advanced features. 

Also, we will try both ways with encoding-aware features and without it. It will be interesting to see how much character encoding contributes to performance, and how much performance could be improved without depending on it.

## Evaluation
Identifying a corpus to belong to a specific language is a hard task. The reason is that many languages like hindi and sanskrit have similar character set. So, if we have a corpus to identify to which language it belongs, we might have the option of classifying it to a language set rather than a specific language.

But thanks to Wikipedia, we have annotated corpus. Hence, evaluation can be done mechanically. The metric will be a precision/recall for each language. If our method works as expected, then the classification performance for confusing languages should be improved.
  
## Data set
There are many datasets available for different languages in the internet. The language identification task will depend on the depth of content these datasets will have.

One specific example is Wikipedia, which provides a dataset that consists of 50 languages. In order to make this problem interesting, we will split each document by sentence/paragraph in order to yield shorter documents. We believe that with this number of languages and shorter length of documents, this problem provides sufficient amount of ambiguity that we need.

## Related works
  * [Spoken Language Recognition: From Fundamentals to Practice](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6451097)
  * [Language identification in the limit](http://www.sciencedirect.com/science/article/pii/S0019995867911655)
  * [Language Identification—A Brief Review](http://www.springer.com/cda/content/document/cda_downloaddocument/9783319177243-c2.pdf?SGWID=0-0-45-1503095-p177335300)
