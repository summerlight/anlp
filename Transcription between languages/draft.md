# Transcription Between Languages
## Motivation 

Learning second language has become a trend in the world. It's difficult for second language learner to remember the word or sentence if he or she doesn't pronunce correctly. One way to learn how to pronunce is to look at phonetic notation such as International Phonetic Alphabet(IPA). The other way is to read the script in our native language. This is what we want to focus on, the so-called Orthographic Transcription.

## Existing applications

Most transcription applications are talking about speech to text. We saw an application [Hangulize](http://hangulize.org/) last week. It is open source on Github.

## Method

1. get pronunciation/audio/waveform for each word
2. compare the waveform with that of target language for every unit of sound
3. assign word or character which has highest similarity

## Evaluation

A possible evaluation is that some team members go through the script in target language and give a score on how similar does it sound like in both language. We might both have to know source language and target language or we compare the sound pronunced by some applications, such as Google Translate.

## Data set

The data we like to transcribe could be from some corpus collecting sentences and words in journal, newspapers and academic conversations: [example](http://corpus.byu.edu/).

## Related works
  * [Automatic Alignment and Analysis of Linguistic Change - Transcription Guidelines](http://fave.ling.upenn.edu/downloads/Transcription_guidelines_FAAV.pdf)
  * [The ESTER 2 Evaluation Campaign for the Rich Transcription of French Radio Broadcasts](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.332.3734&rep=rep1&type=pdf)
  * [Steps in Doing a Transcription](http://www.kcl.ac.uk/sspp/departments/education/research/ldc/knowledge-transfer/DATA/part3.pdf)
