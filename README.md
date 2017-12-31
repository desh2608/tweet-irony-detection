# tweet-irony-detection

## Setup
* Download the GloVe vectors trained on 2B tweets from [here](http://nlp.stanford.edu/data/glove.twitter.27B.zip) and save the 100-D vectors in a firectory named __glove__
* The __datasets__ directory contains preprocessed data for Task A. The *main.py* uses this preprocessed data.
* For transfer learning, we need to use the __SARC__ dataset available [here](http://nlp.cs.princeton.edu/SARC/2.0/main/). Just download the *comments.json* and *train-balanced* files. Preprocessing code is available in __datasets/SARC/__ 

## Description
* Each tweet consists of 2 parts. The first part is the text (with emoji), and the second is the hashtag.
* We conjecture that the presence of irony in a tweet may be detected by the relation between the tweet text and the hashtags. For instance, "I'm so happy with this result #not #fuckthis" contains irony because the semantics of the text and the hashtags are opposite.
* Following this conjecture, we model the problem of irony detection as a relation classification problem, where the 2 entities are the text (+emoji) representation and the hashtag representation.
* The text is fed into a Bi-LSTM (let's call this LSTM 1) to get a d1 dimensional vector. The presence/absence of top 10 emojis is represented in a d2 dimensional vector. These are concatenated to get the text representation.
* Hashtags are also fed into a separate LSTM (LSTM 2) to get a d1+d2 dimensional vector.
* Relation between the vectors is computed using __circular correlation__ method (also called Holographic embeddings, see the 3rd subsection [here](https://medium.com/explorations-in-language-and-learning/beyond-euclidean-embeddings-c125bbd07398)).
* Finally a softmax layer performs this computation.

## Results and further work
* Currently the system gives ~61% F-score on Task 1 after hyperparameter tuning. I haven't yet run it on Task 2.
* The major issue is that a lot of tweets don't have hashtags, in which case, we need to infer irony only from text. Since the training data is relatively small, learning is difficult.
* We need to train *LSTM 1* on the SARC dataset, which contains labeled data (0/1) of whether a comment contains irony. This should improve performance at least in Task 1.
* Also, I'm not sure if an LSTM is the best way to obtain vectors for hashtags since order is not important in that case. But first let's start with the transfer learning part.