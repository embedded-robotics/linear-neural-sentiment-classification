# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from sentiment_data import Counter, List
from utils import *
from collections import Counter
from nltk.corpus import stopwords

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        # Firstly add the vocabulary to indexer if add_to_indexer is True
        if add_to_indexer:
            for word in sentence:
                self.indexer.add_and_get_index(word)

        # Now extract the features using the counter
        return Counter(sentence)

    def get_feature_counter_cache(self, train_exs: List[SentimentExample]) -> List[Counter]:
        feature_counter_list = []
        feature_labels_list = []

        for train_sample in train_exs:
            word_list = train_sample.words
            sample_label = train_sample.label

            # Convert to lower case
            word_list = [word.lower() for word in  word_list]

            # Extracting the features for the single sample and building the dictionary
            sample_features = self.extract_features(word_list, add_to_indexer=True)

            # Storing the feature counters and sample labels
            feature_counter_list.append(sample_features)
            feature_labels_list.append(sample_label)

        return (feature_counter_list, feature_labels_list)


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:

        # Creating the biword list
        biword_list = []
        for i in range (0, len(sentence)-1):
            biword_list.append(sentence[i] + " " + sentence[i+1])

        # Firstly add the vocabulary to indexer if add_to_indexer is True
        if add_to_indexer:
            for word in biword_list:
                self.indexer.add_and_get_index(word)

        # Now extract the features using the counter
        return Counter(biword_list)

    def get_feature_counter_cache(self, train_exs: List[SentimentExample]) -> List[Counter]:
        feature_counter_list = []
        feature_labels_list = []

        for train_sample in train_exs:
            word_list = train_sample.words
            sample_label = train_sample.label

            # Convert to lower case
            word_list = [word.lower() for word in  word_list]

            # Extracting the features for the single sample and building the dictionary
            sample_features = self.extract_features(word_list, add_to_indexer=True)

            # Storing the feature counters and sample labels
            feature_counter_list.append(sample_features)
            feature_labels_list.append(sample_label)

        return (feature_counter_list, feature_labels_list)


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:

        # Creating the biword list
        biword_list = []
        for i in range (0, len(sentence)-1):
            biword_list.append(sentence[i] + " " + sentence[i+1])

        # Creating the triword list
        triword_list = []
        for i in range (0, len(sentence)-2):
            triword_list.append(sentence[i] + " " + sentence[i+1] + " " + sentence[i+2])

        # Creating the combined list
        combined_list = sentence + biword_list + triword_list

        # Firstly add the vocabulary to indexer if add_to_indexer is True
        if add_to_indexer:
            for word in combined_list:
                self.indexer.add_and_get_index(word)

        # Now extract the features using the counter
        return Counter(combined_list)

    def get_feature_counter_cache(self, train_exs: List[SentimentExample]) -> List[Counter]:
        feature_counter_list = []
        feature_labels_list = []

        punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        stop_words =  list(stopwords.words("english"))

        for train_sample in train_exs:
            word_list = train_sample.words
            sample_label = train_sample.label

            # Convert to lower case
            word_list = [word.lower() for word in  word_list]

            # Remove Stop Words
            word_list = [word for word in  word_list if word not in stop_words]

            # Remove all the individual puncutation characters
            modified_word_list = []
            for word in word_list:
                for letter in word:
                    if letter in punctuation:
                        word = word.replace(letter, "")
                modified_word_list.append(word)

            # Remove all the strings having length count less than 3
            word_list = [word for word in modified_word_list if len(word) > 3]

            # Extracting the features for the single sample and building the dictionary
            sample_features = self.extract_features(word_list, add_to_indexer=True)

            # Storing the feature counters and sample labels
            feature_counter_list.append(sample_features)
            feature_labels_list.append(sample_label)

        return (feature_counter_list, feature_labels_list)

def logistic_function(feature_vector, weights_vector):
    dot_product = np.dot(feature_vector, weights_vector)
    prob_y = np.exp(dot_product)/(np.exp(dot_product) + 1)
    return prob_y

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weight_vector : np, feat_extractor: FeatureExtractor):
        self.weight_vector = weight_vector
        self.feat_extractor = feat_extractor

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        # Get the counter for a sample
        feature_counter = self.feat_extractor.extract_features(ex_words)

        # Get all the objects of the indexer
        indexer = self.feat_extractor.get_indexer()

        # Now form a feature list
        object_index = []
        for key in feature_counter.keys():
            object_index.append(indexer.index_of(key))

        # Make a feature vector from a feature list
        feature_vector = np.zeros(len(indexer))
        feature_vector[[object_index]] = 1

        # Calculating the logistic function using feature_vector and weights_vector
        logistic_out = logistic_function(feature_vector, self.weight_vector)

        # Defining the boundary using the logistic out
        if logistic_out >= 0.5:
            return 1
        else:
            return 0


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    # Shuffle all the training examples before getting their features
    random.seed(42)
    random.shuffle(train_exs)

    # Get a feature cache for all the samples in form of a Counter list
    features_counter, features_label = feat_extractor.get_feature_counter_cache(train_exs)

    # Get all the objects of the indexer
    indexer = feat_extractor.get_indexer()

    # Initialize the weight vector and feature vector
    weight_vector = np.zeros(len(indexer))
    feature_vector = np.zeros(len(indexer))

    # Initializing the learning rate and epochs
    learning_rate = 0.1
    epochs = 20

    # Iterate over each instance of the features and form the feature vector on the fly
    for i in range(epochs):
        for (counter, label) in zip(features_counter, features_label):

            object_index = []
            for key in counter.keys():
                object_index.append(indexer.index_of(key))

            # Zero out the feature vector before populating the positions with 1 for features
            feature_vector.fill(0)
            feature_vector[[object_index]] = 1

            # Calculating the logistic function using feature_vector and weights_vector
            logistic_out = logistic_function(feature_vector, weight_vector)

            # Calculate the gradient to update the weight vectors
            gradient = feature_vector * (label - logistic_out)

            # Update the weights for the next sample
            weight_vector = weight_vector + learning_rate*gradient

    # Form an instance of logistic regression classifer and return it
    logistic_regression_classifier = LogisticRegressionClassifier(weight_vector, feat_extractor)
    return logistic_regression_classifier

def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model

class FFNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_layers, output_classes, word_embeddings):
        super(FFNN_LSTM, self).__init__()
        self.embed = word_embeddings.get_initialized_embedding_layer()
        self.lstm = nn.LSTM(input_size, hidden_layers, 1, batch_first=True)
        self.linear1 = nn.Linear(hidden_layers, 64)
        # self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64, output_classes)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, input_data):
        x = self.embed(input_data)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.linear1(x)
        # x = self.relu1(x)
        x = self.linear2(x)
        return x


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_classes, word_embeddings):
        super(FFNN, self).__init__()
        self.embed = word_embeddings.get_initialized_embedding_layer()
        self.linear1 = nn.Linear(input_size, hidden_layers)
        # self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_layers, 64)
        # self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(64, output_classes)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, input_data):
        x = self.embed(input_data)
        x = torch.mean(x, dim=1)
        x = self.linear1(x)
        # x = self.relu1(x)
        x = self.linear2(x)
        # x = self.relu2(x)
        x = self.linear3(x)
        return x

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        self.network = network
        self.word_embeddings = word_embeddings

    def predict(self, ex_words: List[str], pad_length = 40) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """

        # Storing the word indices from the example words
        word_indices = []
        for word in ex_words:
            word_index = self.word_embeddings.word_indexer.index_of(word)
            if word_index != -1:
                word_indices.append(word_index)
            else:
                word_indices.append(1) # 'UNK' index

        # Pad or truncate the index
        index_length = len(word_indices)
        if index_length > pad_length:
            word_indices = word_indices[0:pad_length]
        else:
            for i in range(index_length, pad_length):
                word_indices.append(0) # 'PAD' index

        # Converting the indices to tensor input
        x = torch.from_numpy(np.array([word_indices])).int()

        # Getting the output of the model
        output = self.network(x)

        # Getting the argument for highest probability values
        y_pred = torch.argmax(output).item()

        # Returning the predicted output
        return y_pred


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """

    # Chanage the pad length in the predict function definition (default value) if you are changing it here as well
    pad_length=40
    num_epochs = 20
    input_size = 300
    hidden_layers = 128
    output_classes = 2
    batch_size = 1
    total_samples = len(train_exs)
    batch_indices = np.arange(0, total_samples, batch_size)
    initial_learning_rate = 0.001

    ffnn = FFNN(input_size, hidden_layers, output_classes, word_embeddings)
    # ffnn = FFNN_LSTM(input_size, hidden_layers, output_classes, word_embeddings)
    optimizer = optim.Adam(ffnn.parameters(), lr=initial_learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        random.seed(42)
        random.shuffle(train_exs)

        for i in range(0, len(batch_indices)-1):
            train_batch = train_exs[i:i+1]
            word_indices = []
            labels = []
            # Pad or truncate each sample in the current batch
            for train_sample in train_batch:
                word_list = train_sample.words
                index = []
                for word in word_list:
                    word_index = word_embeddings.word_indexer.index_of(word)
                    if word_index != -1:
                        index.append(word_index)
                    else:
                        index.append(1) # 'UNK' index

                # Pad or truncate the index
                index_length = len(index)
                if index_length > pad_length:
                    index = index[0:pad_length]
                else:
                    for i in range(index_length, pad_length):
                        index.append(0) # 'PAD' index

                word_indices.append(index)
                labels.append([train_sample.label])

            x = torch.from_numpy(np.array(word_indices)).int()

            # # Build one-hot representation of y. Instead of the label 0 or 1, y_onehot is either [0, 1] or [1, 0]. This
            # # way we can take the dot product directly with a probability vector to get class probabilities.
            y_onehot = torch.zeros((len(labels), output_classes))
            # # scatter will write the value of 1 into the position of y_onehot given by y
            y_onehot.scatter_(1, torch.from_numpy(np.asarray(labels, dtype=np.int64)), 1)

            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            optimizer.zero_grad()
            output = ffnn(x)
            # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
            loss = criterion(output, y_onehot)
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    neural_sentiment_classifier = NeuralSentimentClassifier(ffnn, word_embeddings)
    return neural_sentiment_classifier