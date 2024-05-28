# Linear and Neural Sentiment Classification

This repo deals with the development of linear classifier and a neural network for a sentiment classification task. Standard pipeline is used as in many NLP tasks like `Data Reading`, `Pre-processing`, `Training`, and `Evaluation`.

* Disclaimer: This work was done as part of the course `CS388: Natural Language Processing` by `Greg Durrett` at Department of Computer Science, UT Austin

1. The dataset used is the movie review snippets taken from Rotten Tomatoes by Scoher et al. (2013). It can be downloaded from Kaggle (https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews).
2. The dataset labels are "fine-grained" with the following labels: `highly negative`, `negative`, `neutral`, `positive` and `highly positive`. For this task, we only consider simplified version of binary sentiment classification (positive/negative) with neutral sentences discarded from the data
3. A Logitic Regression classifier is implemented from **scratch** rather than using **scikit-learn** to solve the task of binary classification. Text features are extracted using bag-of-word unigram/bigram featurization
4. A Deep Averaging Neural Network is implemented using PyTorch for the binary sentiment classification task. `Glove` vectors are used to extract numerical embeddings from the text. These embeddings are then fed into the neural network using Word Embeddings layer (`nn.Embedding`).