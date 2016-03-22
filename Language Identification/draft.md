## Motivation 
  
Usage of interntet is growing at an exponential rate and so does the number of users. These users are from different parts of the world and speak different languages. The content of a page on a website written by an user from one part of the world will not be understood by an user of the website from a different part of the world. The main problem of not being able to understand the web page is due to different languages being used. If an author writes a web page about a topic in his native language and if users from other parts of the world need to understand it, they should first be able to identify the language in which the web page is written.

## Problem definition

The problem which we are targeting involves identifying a language based on the corpus we get. We need to take into consideration, the usage of words or sentence and the way these words or sentences are related to each other from the training corpus.

Define different classes (languages) based on the training corpus.

## Method
We will collect the training corpus of different languages. We consider each language as a class. Apply Naive Bayes to learn the model based on the training corpus. Use this learned model to classify the development corpus.

## Evaluation
Identifying a corpus to belong to a specific language is a hard task. The reason is that many languages like hindi and sanskrit have similar character set. So, if we have a corpus to identify to which language it belongs, we might have the option of classifying it to a language set rather than a specific language.
  
## Data set
There are many datasets available for different languages in the internet. The language identification task will depend on the depth of content these datasets will have.

## Related works
  * [Spoken Language Recognition: From Fundamentals to Practice](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6451097)
  * [Language identification in the limit](http://www.sciencedirect.com/science/article/pii/S0019995867911655)
  * [Language Identification—A Brief Review](http://www.springer.com/cda/content/document/cda_downloaddocument/9783319177243-c2.pdf?SGWID=0-0-45-1503095-p177335300)
