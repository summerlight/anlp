# Transcription Between Languages
## Motivation 

Learning second language has become a trend in the world. It's difficult for second language learner to remember the word or sentence if he or she doesn't pronunce correctly. One way to learn how to pronunce is to look at phonetic notation such as International Phonetic Alphabet(IPA). The other way is to read the script in our native language. This is what we want to focus on, the so-called Orthographic Transcription.

## Problem Definition

  * Input: Words, sentences, paragraphs or articles from corpus in source language.
  * Output: Scripts that are very similar to the sound of input in target language. Note that the output have to follow the orthography of the target language. 

## Existing applications

Most transcription applications are talking about speech to text. We saw an application [Hangulize](http://hangulize.org/) last week. It is open source on Github.

## Method

1. ~~get pronunciation/audio/waveform/phoneme for each word from corpus(see Dataset)~~
2. look up phoneme corpus and get phonemes for every single word
2. compare phoneme with that of target language for each phoneme
3. assign word or character which has highest similarity 

## Evaluation

A possible evaluation is that some team members go through the script in target language and give a score on how similar does it sound like in both language. Therefore, we might both have to know source language and target language or we compare the sound pronunced by some applications, such as Google Translate.

The above might be considered as too subjective so that it's not a good evaluation metric. A more objective metric could be that comparing IPA for each pair of word and corresponding scripts. More similar they are, more successful the method is.

Baseline: 
The script is assigned by choosing the smallest edit distance of IPA

## Data set
### General Corpus
The data we like to transcribe could be from some corpus collecting sentences and words in journal, newspapers and academic conversations: [example](http://corpus.byu.edu/).

### Phoneme Corpus
The corpus are all in English so far.

[TIMIT Acoustic-Phonetic Continuous Speech Corpus](https://catalog.ldc.upenn.edu/LDC93S1):
time-aligned orthographic, phonetic and word transcriptions as well as a 16-bit, 16kHz speech waveform file for each utterance

[CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict):
contains over 134,000 words and their pronunciations

## Related works
  * [Automatic Alignment and Analysis of Linguistic Change - Transcription Guidelines](http://fave.ling.upenn.edu/downloads/Transcription_guidelines_FAAV.pdf)
  * [The ESTER 2 Evaluation Campaign for the Rich Transcription of French Radio Broadcasts](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.332.3734&rep=rep1&type=pdf)
  * [Steps in Doing a Transcription](http://www.kcl.ac.uk/sspp/departments/education/research/ldc/knowledge-transfer/DATA/part3.pdf)
