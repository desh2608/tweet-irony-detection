# Irony detection in tweets

## Setup
* Download the GloVe vectors trained on 2B tweets from [here](http://nlp.stanford.edu/data/glove.twitter.27B.zip) and save the 100-D vectors in a firectory named __glove__

## Feature generation
* Features are obtained from 2 sources:
1. *Holographic embeddings*: Circular cross-correlation between tweet text and hashtag vectors. Use __holographic.ipynb__ for this.
2. *DeepMoji*: This is forked from [here](https://github.com/bfelbo/DeepMoji), and modified to support Python 3.5+. Place __deepmoji_features.ipynb__ in the __DeepMoji/examples__ directory and run to generate features.

## Classification
We use the Python XGBoost package to perform classification based on features obtained as above. Use __xgb_classifier.ipynb__ for this purpose.