## Motivation 
  
  Improving signal-to-noise ratio of social media data becomes increasingly important. One source of noise is inherent ambiguity of natural language, and a short nature of social media text accelerates this problem. For instance of English, when if we met a noun "python", does it indicate a programming language or a snake?

  Thanks to graphical structures of social media data, there are lots of potential to exploit this social relationships to reveal valuable information for natural language processing, or inversely, NLP processing could be used for reinforcing this structure. For exmaple, if there is an account which likes technical computing texts and it mentioned "python", then it is much more likely to be a programming language.

## Problem definition

  Currently, we consider this problem as a kind of topic modeling problem. Also, we do not expect our data set to have any NLP-related annotation, we can further refine the problem to a unsupervised one.
  
  Word sense disambiguation is another possible approach. But since our focus is more about improving signal-to-noise ratio than exact textual search, we expect topic modeling should yield better results.
  
  An average length of social media text is not sufficient to reliably determine its topic. In order to overcome this problem, we intend to actively exploit social relationships. Hence, the problem we want to solve is encoding this side information into existing framework, LDA in this case.

## Method
  We will use Latent Dirichlet Allocation as a starting point. Based on this method, we could experiment various ways to exploit the social structure and improve the result.
  The basic step is briefly:
  1. Collect corpus from Twitter
  1. Implement LDA and test it on the corpus
  1. Develop (or apply existing) a method to encode graph information into LDA framework : This will be our own contribution.
  1. Compare the result

  One of the ideas for encoding graph information is that a social network contains variety of communities, and we expect that there should be correlation between each community and a topic distribution. If we can cluster a social network by communities and develop a method that represent this intuition as a prior knowledge, we could apply a different (and hopefully more accurate) model to each text, depending on the context.
  
  Note that detailed methodology could be drastically changed during this project. For instance, Latent Semantic Indexing is another valid candidate for this research. We choose LDA because there are more resources and researchs out there on the internet. 

## Evaluation
  This is an open question, since signal-to-noise is a highly subjective matter. What makes it worse is that our data set won't have annotations. Hence, evaluation should involve a certain amount of human intervention.
  
  A baseline will be a naive application of LDA algorithm to the data set, which is unaware of graph information. 

## Data set
  Most of academic data sets for social networks are no more publicly available, mainly due to the request from the data owner. Moreover, most social network service does not open their data to the public. Hence, we will crawl Twitter by using its API. This data set may contain below information:

  - Tweet : Text, User, Geographic location, Posting time, Hash tag, URLs, User mentions...
  - User to tweet : Retweeted, Liked, Posted
  - Tweet to tweet : Replying, Referred
  - User to user : Following, List, Conversations ...

  This data set does not have any annotation and is expected to be large, so it effectively limits our approach to unsupervised learning. Therefore, the languages we will use are limited to those languages that we can understand.
  One open question is how to crawl; Twitter API imposes some constraints on an API request rate, so crawling every data that we can access would not be a wise way.

## Related works
  * [Community detection in graphs](http://arxiv.org/pdf/0906.0612v2.pdf)
  * [Empirical Comparison of Algorithms for Network Community Detection](https://cs.stanford.edu/people/jure/pubs/communities-www10.pdf)
  * [Detecting Community Kernels in Large Social Network](http://www.cs.cornell.edu/~lwang/Wang11ICDM.pdf)
  * [Rethinking LDA: Why Priors Matter](http://mimno.infosci.cornell.edu/papers/NIPS2009_0929.pdf)
  * [topic models: priors, stop words and languages](http://people.cs.umass.edu/~wallach/talks/priors.pdf)
  * [Word Features for Latent Dirichlet Allocation](http://www.tiberiocaetano.com/papers/2010/PetSmoCaeBunetal10.pdf) - This paper extends LDA to allow for the encoding of side information.