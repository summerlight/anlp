## Motivation 
  
Filtering out uninteresting noise from social media becomes increasingly important. One source of noise is inherent ambiguity of natural language, For instance of English, when if we met a noun "python", does it indicate a programming language or a snake? Topic models could be a useful tool for disambiguating such situation. However, usual social media text is too short to provide meaningful information without the context.

Thanks to graphical structures of social media data, there are lots of potential to exploit this social relationships to reveal valuable information for natural language processing. For exmaple, if there is an account which likes technical computing texts and it mentioned "python", then it is much more likely to be a programming language. We'd like to investigate this opportunity.

## Problem definition

Currently, we consider this problem as a unsupervised topic modeling problem since our data set to have any NLP annotation.
  
In this case, an average length of social media text is not sufficient to reliably determine its topic. To overcome this problem, we will exploit social relationships. Hence, the problem we want to solve is encoding this side information into existing framework, LDA in this case.

If this method works well, we expect this method will yield higher-quality, context-sensitive topic models. (We know that this is ambiguous; evaluation is a really hard problem.)

## Method
We will use Latent Dirichlet Allocation as a starting point. Based on this method, we could experiment various ways to exploit the social structure and improve the result.

The basic step is briefly:
  1. Collect corpus from Twitter
  1. Implement LDA and test it on the corpus
  1. Develop (or apply existing) a method to encode graph information into LDA framework : This will be our own contribution.
  1. Compare the result

One of the ideas for using graph information is that a social network contains variety of communities, and we expect that there should be correlation between each community and a topic distribution. If we can cluster a social network by communities and develop a method that represent this intuition as a prior knowledge, we could apply a different (and hopefully more accurate) model to each text, depending on the context.

The main challenge is how to encode social relationships. We could adjust hyperparameters of LDA, modify LDA itself to cover this side information, or directly append some "hint words" inferred from social relationships to documents. We will experiment various ways and choose the the best method.

Note that detailed methodology could be drastically changed during this project. For instance, Latent Semantic Indexing or Pachinko Allocation is another valid candidate for this research. We choose LDA because it is the simplest and there are more resources/researchs out there on the internet. 

## Evaluation
Topic models are known to be a hard problem to evaluate. This problem could be even harder because:

  * Since we want to use not only the text, but also its context. This effectively invalidates many assumptions made by typical evaluation methods.
  * Our data set won't have annotations. Hence, evaluation should involve a certain amount of human intervention.

Our initial search for a good evaluation method does not yield easy results for our purpose. In the worst case, manual evaluation is inevitable, then we will sample thousands of tweets and see if they made sense.

A baseline will be a naive application of LDA algorithm to the same data set, which is unaware of social relationships.

## Data set
Most of academic data sets for social networks are no more publicly available, mainly due to the request from the data owner. Moreover, most social network service does not open their data to the public. Hence, we will crawl Twitter by using its API. This data set may contain below information:

  - Tweet : Text, User, Geographic location, Posting time, Hash tag, URLs, User mentions...
  - User to tweet : Retweeted, Liked, Posted
  - Tweet to tweet : Replying, Referred
  - User to user : Following, List, Conversations ...

This data set does not have annotations and is expected to be large, so it effectively limits our approach to unsupervised learning. Therefore, the languages we will use are limited to those languages that we can understand.

For raw tweet texts, we will do basic tokenization, POS tagging and filter out uninformative tokens. For Korean, we expect a KoNLPy package to do the most of dirty jobs for us. 

One open question is how to crawl; Twitter API imposes some constraints on an API request rate, so crawling every data that we can access would not be a wise way.

## Related works
### Graph clustering
  * [Community detection in graphs](http://arxiv.org/pdf/0906.0612v2.pdf)
  * [Empirical Comparison of Algorithms for Network Community Detection](https://cs.stanford.edu/people/jure/pubs/communities-www10.pdf)
  * [Detecting Community Kernels in Large Social Network](http://www.cs.cornell.edu/~lwang/Wang11ICDM.pdf)

### Attempt to understand/modify LDA
  * [Rethinking LDA: Why Priors Matter](http://mimno.infosci.cornell.edu/papers/NIPS2009_0929.pdf)
  * [topic models: priors, stop words and languages](http://people.cs.umass.edu/~wallach/talks/priors.pdf)
  * [Word Features for Latent Dirichlet Allocation](http://www.tiberiocaetano.com/papers/2010/PetSmoCaeBunetal10.pdf) - This paper extends LDA to allow for the encoding of side information.

### How to evaluate topic models
  * [Reading Tea Leaves: How Humans Interpret Topic Models](http://www.umiacs.umd.edu/~jbg/docs/nips2009-rtl.pdf) 
  * [Evaluation Methods for Topic Models](http://mimno.infosci.cornell.edu/papers/wallach09evaluation.pdf)
  * [External Evaluation of Topic Models: A Graph Mining Approach](http://www3.cs.stonybrook.edu/~leman/pubs/13-icdm-topics-cameraready.pdf)
  * [What are good ways of evaluating the topics generated by running LDA on a corpus?](https://www.quora.com/What-are-good-ways-of-evaluating-the-topics-generated-by-running-LDA-on-a-corpus)
  * [Termite: Visualization Techniques for Assessing Textual Topic Models](http://vis.stanford.edu/papers/termite)