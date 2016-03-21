#Lyric Translation using Machine Translation Stochatic Model
##Motivation
In the constantly connected world where more people are listening to music of different genre and mostly in different languages. There is a need for translating lyrics into users language to increase the user perception of meaning in the song. An example for such a song would be the K-Pop hit Gangum Style, though was a great hit in the music market, people did not understand the lyrics of the song is. What if they was a way to convert the lyrics which are in English to other languages and vice versa.

We currently have lingustic experts to convert a document from one language to other. We could see this in various series telecasted over the world. The sub titles for each of the series are manually documented and proof read. Could we achieve this in a more automated way?

##Problem Definition
The problem in hand is machine translation of document from one corpus to another. But lyrics as such do not have a complete meaning in the sentences which makes it hard to understand the context of the words (features) in the document and predict the optimal mapped value in the other corpora.

Word sense disambiguation is another possible approach which might be a good start in solving the problem, but improving the accuracy of the context mapping, we need to implement generative modeling approach to get better results.

An average lyrics would have so many redundant words and sentences which would not be good approach to indicate a strong mapping between bilingual corpus.

##Method
Machine translation (MT) is automated translation. It is the process by which computer software is used to translate a text from one natural language (such as English) to another (such as Spanish).
Translation is not a mere word-for-word substitution. A translator must interpret and analyze all of the elements in the text and know how each word may influence another.

There are two methods of approaching the problem.
*Rule-Based Machine Translation
*Statistical Machine Translation


In a rule based MT system, we would achieve the following using
Steps:
1. Parsing the text and create a transistional representation from which the text in the target language is generated.
2. Learn rule sets and transfer the grammatical structure of the source language into the target language.
3. Create vocabulary and linguistic rules.

A statistical MT system spans across different fields,
1. Syntax-Based SMT
2. Phrase-Based SMT
3. Word Alignment
4. Language Modeling

An example of such model would be as represented in the diagram 
* http://research.microsoft.com/en-us/projects/mt/msrmt1.gif
* http://research.microsoft.com/en-us/projects/mt/msrmt2.gif

####Highlighting the positive and negatives
Rule-Based MT					                    |   Statistical MT
------------------------------------------|-------------------------------------
+ Consistent and predictable quality		  |   – Unpredictable translation quality
+ Out-of-domain translation quality		    |   – Poor out-of-domain quality
+ Knows grammatical rules			            |   – Does not know grammar
+ High performance and robustness		      |   – High CPU and disk space requirements
+ Consistency between versions			      |   – Inconsistency between versions
– Lack of fluency				                  |   + Good fluency
– Hard to handle exceptions to rules		  |   + Good for catching exceptions to rules
– High development and customization costs|  + Rapid and cost-effective development costs provided the required corpus exists

##Evaluation
The evaluation of the system would depend on the accuracy of the translation betwen the source and the target language

##Data set for learning and patch testing
Currently there is a manual site where lingustic expert trans lyrics from one language to another. 
* [Lyric Translate](lyricstranslate.com)
It has about 38000 articles and would be good source of training and development set.

* [AZ Lyric](http://www.azlyrics.com/)
This contains lyrics for new english songs which we can use for testing

##References and Related Works
* [Machine Translation](http://research.microsoft.com/en-us/projects/mt/)
* [Statistical Dependency Graph](http://research.microsoft.com/pubs/68973/stat_mt_dependency_graph_tmi_camera_ready.pdf/)
* [Word Alignment](http://research.microsoft.com/pubs/68848/acl-2001-alignment.doc)
* [Decision Trees](http://research.microsoft.com/pubs/68909/amta-decision-trees.doc)
* [Sentence Training](http://research.microsoft.com/pubs/68968/conf_lrec2004.pdf)
* [Cross Lingual Semantic Relateness](http://web.eecs.umich.edu/~mihalcea/downloads.html#CROSS_LIN_SEM_REL)
