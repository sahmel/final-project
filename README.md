# Artificial Intelligence Nanodegree
## Machine Translation Project


## Introduction
In this notebook, you find a number of deep neural networks that function as part of an end-to-end machine translation pipeline. The pipelines will accept English text as input and return the French translation.



```python
%load_ext autoreload
%aimport helper, tests
%autoreload 1
```


```python
import collections

import helper
import numpy as np
import project_tests as tests

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import LSTM, GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Embedding, Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, rmsprop
from keras.losses import sparse_categorical_crossentropy
```

    Using TensorFlow backend.


### Verify access to the GPU
The following test applies only if you expect to be using a GPU. Run the next cell, and verify that the device_type is "GPU".


```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

    [name: "/cpu:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 1860777044165494850
    , name: "/gpu:0"
    device_type: "GPU"
    memory_limit: 357433344
    locality {
      bus_id: 1
    }
    incarnation: 3563910225959155661
    physical_device_desc: "device: 0, name: Tesla K80, pci bus id: 0000:00:04.0"
    ]


## Dataset
I'll begin by investigating the dataset that will be used to train and evaluate the pipeline.  The most common datasets used for machine translation are from [WMT](http://www.statmt.org/).  However, that will take a long time to train a neural network on.  I'll be using a dataset we created for this project that contains a small vocabulary.  
### Load Data
The data is located in `data/small_vocab_en` and `data/small_vocab_fr`. The `small_vocab_en` file contains English sentences with their French translations in the `small_vocab_fr` file. Load the English and French data from these files from running the cell below.


```python
# Load English data
english_sentences = helper.load_data('data/small_vocab_en')
# Load French data
french_sentences = helper.load_data('data/small_vocab_fr')

print('Dataset Loaded')
```

    Dataset Loaded


### Files
Each line in `small_vocab_en` contains an English sentence with the respective translation in each line of `small_vocab_fr`.  View the first two lines from each file.


```python
for sample_i in range(2):
    print('small_vocab_en Line {}:  {}'.format(sample_i + 1, english_sentences[sample_i]))
    print('small_vocab_fr Line {}:  {}'.format(sample_i + 1, french_sentences[sample_i]))
```

    small_vocab_en Line 1:  new jersey is sometimes quiet during autumn , and it is snowy in april .
    small_vocab_fr Line 1:  new jersey est parfois calme pendant l' automne , et il est neigeux en avril .
    small_vocab_en Line 2:  the united states is usually chilly during july , and it is usually freezing in november .
    small_vocab_fr Line 2:  les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .


From looking at the sentences, you can see they have been preprocessed already.  The puncuations have been delimited using spaces. All the text have been converted to lowercase.  This should save you some time, but the text requires more preprocessing.
### Vocabulary
The complexity of the problem is determined by the complexity of the vocabulary.  A more complex vocabulary is a more complex problem.  Let's look at the complexity of the dataset we'll be working with.


```python
english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')
```

    1823250 English words.
    227 unique English words.
    10 Most common words in the English dataset:
    "is" "," "." "in" "it" "during" "the" "but" "and" "sometimes"
    
    1961295 French words.
    355 unique French words.
    10 Most common words in the French dataset:
    "est" "." "," "en" "il" "les" "mais" "et" "la" "parfois"


For comparison, _Alice's Adventures in Wonderland_ contains 2,766 unique words of a total of 15,500 words.
## Preprocess
For this project, you won't use text data as input to your model. Instead, you'll convert the text into sequences of integers using the following preprocess methods:
1. Tokenize the words into ids
2. Add padding to make all the sequences the same length.

Time to start preprocessing the data...
### Tokenize
For a neural network to predict on text data, it first has to be turned into data it can understand. Text data like "dog" is a sequence of ASCII character encodings.  Since a neural network is a series of multiplication and addition operations, the input data needs to be number(s).

We can turn each character into a number or each word into a number.  These are called character and word ids, respectively.  Character ids are used for character level models that generate text predictions for each character.  A word level model uses word ids that generate text predictions for each word.  Word level models tend to learn better, since they are lower in complexity, so we'll use those.

Turn each sentence into a sequence of words ids using Keras's [`Tokenizer`](https://keras.io/preprocessing/text/#tokenizer) function. Use this function to tokenize `english_sentences` and `french_sentences` in the cell below.

Running the cell will run `tokenize` on sample data and show output for debugging.


```python
def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    
    return tokenizer.texts_to_sequences(x), tokenizer
tests.test_tokenize(tokenize)

# Tokenize Example output
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))
```

    {'the': 1, 'quick': 2, 'a': 3, 'brown': 4, 'fox': 5, 'jumps': 6, 'over': 7, 'lazy': 8, 'dog': 9, 'by': 10, 'jove': 11, 'my': 12, 'study': 13, 'of': 14, 'lexicography': 15, 'won': 16, 'prize': 17, 'this': 18, 'is': 19, 'short': 20, 'sentence': 21}
    
    Sequence 1 in x
      Input:  The quick brown fox jumps over the lazy dog .
      Output: [1, 2, 4, 5, 6, 7, 1, 8, 9]
    Sequence 2 in x
      Input:  By Jove , my quick study of lexicography won a prize .
      Output: [10, 11, 12, 2, 13, 14, 15, 16, 3, 17]
    Sequence 3 in x
      Input:  This is a short sentence .
      Output: [18, 19, 3, 20, 21]


### Padding 
When batching the sequence of word ids together, each sequence needs to be the same length.  Since sentences are dynamic in length, we can add padding to the end of the sequences to make them the same length.

Make sure all the English sequences have the same length and all the French sequences have the same length by adding padding to the **end** of each sequence using Keras's [`pad_sequences`](https://keras.io/preprocessing/sequence/#pad_sequences) function.


```python
def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    
    return pad_sequences(x, maxlen = length, padding = 'post')
tests.test_pad(pad)

# Pad Tokenized output
test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))
```

    Sequence 1 in x
      Input:  [1 2 4 5 6 7 1 8 9]
      Output: [1 2 4 5 6 7 1 8 9 0]
    Sequence 2 in x
      Input:  [10 11 12  2 13 14 15 16  3 17]
      Output: [10 11 12  2 13 14 15 16  3 17]
    Sequence 3 in x
      Input:  [18 19  3 20 21]
      Output: [18 19  3 20 21  0  0  0  0  0]


### Preprocess Pipeline


```python
def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
    preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)
```

    Data Preprocessed
    Max English sentence length: 15
    Max French sentence length: 21
    English vocabulary size: 199
    French vocabulary size: 344


## Split the data into training and test tests


```python
english_sentences_train, english_sentences_test, french_sentences_train, french_sentences_test = train_test_split(english_sentences, french_sentences, test_size=0.2, random_state=42)
```

## Counting and analyzing vocabulary in the training sets 


```python
english_train_words_counter = collections.Counter([word for sentence in english_sentences_train for word in sentence.split()])
french_train_words_counter = collections.Counter([word for sentence in french_sentences_train for word in sentence.split()])

print('{} English words in the training set.'.format(len([word for sentence in english_sentences_train for word in sentence.split()])))
print('{} unique English words in the training set.'.format(len(english_train_words_counter)))
print('10 Most common words in the English training dataset:')
print('"' + '" "'.join(list(zip(*english_train_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words in the training set.'.format(len([word for sentence in french_sentences_train for word in sentence.split()])))
print('{} unique French words in the training set.'.format(len(french_train_words_counter)))
print('10 Most common words in the French training dataset:')
print('"' + '" "'.join(list(zip(*french_train_words_counter.most_common(10)))[0]) + '"')
```

    1458806 English words in the training set.
    227 unique English words in the training set.
    10 Most common words in the English training dataset:
    "is" "," "." "in" "it" "during" "the" "but" "and" "sometimes"
    
    1568964 French words in the training set.
    354 unique French words in the training set.
    10 Most common words in the French training dataset:
    "est" "." "," "en" "il" "les" "mais" "et" "la" "parfois"


## Counting and analyzing vocabulary in the test sets 


```python
english_test_words_counter = collections.Counter([word for sentence in english_sentences_test for word in sentence.split()])
french_test_words_counter = collections.Counter([word for sentence in french_sentences_test for word in sentence.split()])

print('{} English words in the test set.'.format(len([word for sentence in english_sentences_test for word in sentence.split()])))
print('{} unique English words in the test set.'.format(len(english_test_words_counter)))
print('10 Most common words in the English test dataset:')
print('"' + '" "'.join(list(zip(*english_test_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words in the test set.'.format(len([word for sentence in french_sentences_test for word in sentence.split()])))
print('{} unique French words in the test set.'.format(len(french_test_words_counter)))
print('10 Most common words in the French test dataset:')
print('"' + '" "'.join(list(zip(*french_test_words_counter.most_common(10)))[0]) + '"')
```

    364444 English words in the test set.
    227 unique English words in the test set.
    10 Most common words in the English test dataset:
    "is" "," "." "in" "it" "during" "the" "but" "and" "sometimes"
    
    392331 French words in the test set.
    338 unique French words in the test set.
    10 Most common words in the French test dataset:
    "est" "." "," "en" "il" "les" "mais" "et" "la" "parfois"



```python
preproc_english_sentences_train, preproc_french_sentences_train, english_tokenizer_train, french_tokenizer_train =\
    preprocess(english_sentences_train, french_sentences_train)
    
max_english_sequence_length_train = preproc_english_sentences_train.shape[1]
max_french_sequence_length_train = preproc_french_sentences_train.shape[1]
english_vocab_size_train = len(english_tokenizer_train.word_index)
french_vocab_size_train = len(french_tokenizer_train.word_index)

print('Training Data Preprocessed')
print("Max English train dataset sentence length:", max_english_sequence_length_train)
print("Max French train dataset sentence length:", max_french_sequence_length_train)
print("English train dataset vocabulary size:", english_vocab_size_train)
print("French train dataset vocabulary size:", french_vocab_size_train)
print("English train dataset shape:", preproc_english_sentences_train.shape)
print("French train dataset shape:", preproc_french_sentences_train.shape)
```

    Training Data Preprocessed
    Max English train dataset sentence length: 15
    Max French train dataset sentence length: 21
    English train dataset vocabulary size: 199
    French train dataset vocabulary size: 343
    English train dataset shape: (110288, 15)
    French train dataset shape: (110288, 21, 1)


## Preprocessing pipeline on the test sets


```python
preproc_english_sentences_test, preproc_french_sentences_test, english_tokenizer_test, french_tokenizer_test =\
    preprocess(english_sentences_test, french_sentences_test)
    
max_english_sequence_length_test = preproc_english_sentences_test.shape[1]
max_french_sequence_length_test = preproc_french_sentences_test.shape[1]
english_vocab_size_test = len(english_tokenizer_test.word_index)
french_vocab_size_test = len(french_tokenizer_test.word_index)

print('Test Data Preprocessed')
print('Max English test dataset sentence length:', max_english_sequence_length_test)
print('Max French test dataset sentence length:', max_french_sequence_length_test)
print('English test datset vocab size:', english_vocab_size_test)
print('French test dataset vocab size', french_vocab_size_test)
print("English test dataset shape:", preproc_english_sentences_test.shape)
print("French test dataset shape:", preproc_french_sentences_test.shape)
```

    Test Data Preprocessed
    Max English test dataset sentence length: 15
    Max French test dataset sentence length: 20
    English test datset vocab size: 199
    French test dataset vocab size 326
    English test dataset shape: (27573, 15)
    French test dataset shape: (27573, 20, 1)


## Models
In this section, you will experiment with various neural network architectures.
You will begin by training four relatively simple architectures.
- Model 1 is a simple RNN
- Model 2 is a RNN with Embedding
- Model 3 is a Bidirectional RNN
- Model 4 is an optional Encoder-Decoder RNN

After experimenting with the four simple architectures, you will construct a deeper architecture that is designed to outperform all four models.
### Ids Back to Text
The neural network will be translating the input to words ids, which isn't the final form we want.  We want the French translation.  The function `logits_to_text` will bridge the gab between the logits from the neural network to the French translation.  You'll be using this function to better understand the output of the neural network.


```python
def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('`logits_to_text` function loaded.')
```

    `logits_to_text` function loaded.


### Model 1: RNN (IMPLEMENTATION)
![RNN](images/rnn.png)
A basic RNN model is a good baseline for sequence data.  In this model, you'll build a RNN that translates English to French.


```python
from keras.models import Sequential
from keras.layers import SimpleRNN, BatchNormalization
```

### RNN Model on Training and Test Data


```python
from keras.models import Sequential
from keras.layers import SimpleRNN, BatchNormalization

def simple_model_split(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Build the layers
    
    model = Sequential()
    model.add(GRU(english_vocab_size, return_sequences=True, input_shape=input_shape[1:]))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    #model.add(Dense(french_vocab_size, activation='softmax'))
    
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(.001), metrics=['acc'])        
    
    return model

tests.test_simple_model(simple_model_split)

# Reshaping the input to work with a basic RNN
tmp_x = pad(preproc_english_sentences_train, max_french_sequence_length_train)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences_train.shape[-2], 1))

test_x = pad(preproc_english_sentences_test, max_french_sequence_length_test)
test_x = test_x.reshape((-1, preproc_french_sentences_train.shape[-2], 1))

preproc_french_sentences_test = preproc_french_sentences_test.reshape((-1, 
                                                                       preproc_french_sentences_train.shape[-2], 1))
```


```python
print('tmp_x shape:', tmp_x.shape)
print('test_x shape:', test_x.shape)
print('preproc french sentences shape:', preproc_french_sentences_test.shape)
```

    tmp_x shape: (110288, 21, 1)
    test_x shape: (26260, 21, 1)
    preproc french sentences shape: (26260, 21, 1)



```python
# Train the neural network
simple_rnn_model_split = simple_model_split(
    tmp_x.shape,
    max_french_sequence_length_train,
    english_vocab_size,
    french_vocab_size)
simple_rnn_model_split.fit(tmp_x, preproc_french_sentences_train, batch_size=1024, epochs=12, 
                           validation_split=0.2)

# Print prediction(s)
print(logits_to_text(simple_rnn_model_split.predict(tmp_x[:1])[0], french_tokenizer_train))
```

    Train on 88230 samples, validate on 22058 samples
    Epoch 1/12
    88230/88230 [==============================] - 8s 89us/step - loss: 2.8535 - acc: 0.4580 - val_loss: 2.1710 - val_acc: 0.5129
    Epoch 2/12
    88230/88230 [==============================] - 7s 83us/step - loss: 1.8883 - acc: 0.5572 - val_loss: 1.6774 - val_acc: 0.5804
    Epoch 3/12
    88230/88230 [==============================] - 7s 84us/step - loss: 1.5780 - acc: 0.5938 - val_loss: 1.4970 - val_acc: 0.6021
    Epoch 4/12
    88230/88230 [==============================] - 7s 83us/step - loss: 1.4397 - acc: 0.6108 - val_loss: 1.3893 - val_acc: 0.6160
    Epoch 5/12
    88230/88230 [==============================] - 7s 83us/step - loss: 1.3500 - acc: 0.6254 - val_loss: 1.3129 - val_acc: 0.6324
    Epoch 6/12
    88230/88230 [==============================] - 7s 83us/step - loss: 1.2792 - acc: 0.6382 - val_loss: 1.2484 - val_acc: 0.6420
    Epoch 7/12
    88230/88230 [==============================] - 7s 83us/step - loss: 1.2169 - acc: 0.6492 - val_loss: 1.1938 - val_acc: 0.6532
    Epoch 8/12
    88230/88230 [==============================] - 7s 83us/step - loss: 1.1662 - acc: 0.6602 - val_loss: 1.1472 - val_acc: 0.6631
    Epoch 9/12
    88230/88230 [==============================] - 7s 83us/step - loss: 1.1241 - acc: 0.6693 - val_loss: 1.1071 - val_acc: 0.6733
    Epoch 10/12
    88230/88230 [==============================] - 7s 83us/step - loss: 1.0871 - acc: 0.6783 - val_loss: 1.0713 - val_acc: 0.6813
    Epoch 11/12
    88230/88230 [==============================] - 7s 83us/step - loss: 1.0537 - acc: 0.6873 - val_loss: 1.0409 - val_acc: 0.6894
    Epoch 12/12
    88230/88230 [==============================] - 7s 83us/step - loss: 1.0273 - acc: 0.6934 - val_loss: 1.0167 - val_acc: 0.6981
    la est est généralement le en en mais il est jamais agréable agréable en <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>


### Simple RNN Model Scoring


```python

simple_rnn_model_score = simple_rnn_model_split.evaluate(test_x, preproc_french_sentences_test, verbose=0)

print("Simple RNN model accuracy on test dataset: {0:.2f}%".format(simple_rnn_model_score[1]*100))
```

    Simple RNN model accuracy on test dataset: 38.16%


### Model 2: Embedding 
![RNN](images/embedding.png)
An embedding is a vector representation of the word that is close to similar words in n-dimensional space, where the n represents the size of the embedding vectors.

### Embedding Model on Training and Test Data


```python
from keras.utils.vis_utils import plot_model

def embed_model_split(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a RNN model using word embedding on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement
    #input_dim = english_vocab_size
    #output_dim = output_sequence_length
    
    model = Sequential()
    #model.add(Embedding(input_dim=english_vocab_size, output_dim=output_sequence_length, input_length=input_shape[1:][0]))
    model.add(Embedding(input_dim=english_vocab_size, output_dim=output_sequence_length, input_shape=input_shape[1:]))
#     model.add(Flatten())
    model.add(GRU(english_vocab_size, return_sequences=True))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    model.add(LSTM(french_vocab_size, return_sequences=True))

    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(.001), metrics=['acc'])        
    model.summary()
    return model

tests.test_embed_model(embed_model_split)


# TODO: Reshape the input
tmp_x = pad(preproc_english_sentences_train, max_french_sequence_length_train)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences_train.shape[-2]))
#works with the reshape commented out
print(tmp_x.shape)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 21, 21)            4179      
    _________________________________________________________________
    gru_4 (GRU)                  (None, 21, 199)           131937    
    _________________________________________________________________
    time_distributed_4 (TimeDist (None, 21, 344)           68800     
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 21, 344)           948064    
    =================================================================
    Total params: 1,152,980
    Trainable params: 1,152,980
    Non-trainable params: 0
    _________________________________________________________________
    (110288, 21)



```python
# TODO: Train the neural network
simple_embed_model_split = embed_model_split(
    tmp_x.shape,
    max_french_sequence_length_train + 1,
    english_vocab_size_train + 1,
    french_vocab_size_train + 1)

    # TODO: Print prediction(s)
simple_embed_model_split.fit(tmp_x, preproc_french_sentences_train, batch_size=1024, epochs=12, validation_split=0.2)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_3 (Embedding)      (None, 21, 22)            4400      
    _________________________________________________________________
    gru_5 (GRU)                  (None, 21, 200)           133800    
    _________________________________________________________________
    time_distributed_5 (TimeDist (None, 21, 344)           69144     
    _________________________________________________________________
    lstm_3 (LSTM)                (None, 21, 344)           948064    
    =================================================================
    Total params: 1,155,408
    Trainable params: 1,155,408
    Non-trainable params: 0
    _________________________________________________________________
    Train on 88230 samples, validate on 22058 samples
    Epoch 1/12
    88230/88230 [==============================] - 22s 253us/step - loss: 5.5731 - acc: 0.4058 - val_loss: 4.4124 - val_acc: 0.4135
    Epoch 2/12
    88230/88230 [==============================] - 21s 243us/step - loss: 3.8782 - acc: 0.4626 - val_loss: 3.3794 - val_acc: 0.4707
    Epoch 3/12
    88230/88230 [==============================] - 21s 243us/step - loss: 3.2711 - acc: 0.4579 - val_loss: 3.1314 - val_acc: 0.4757
    Epoch 4/12
    88230/88230 [==============================] - 21s 243us/step - loss: 2.9257 - acc: 0.4865 - val_loss: 2.7958 - val_acc: 0.4868
    Epoch 5/12
    88230/88230 [==============================] - 21s 243us/step - loss: 2.7797 - acc: 0.4791 - val_loss: 2.6540 - val_acc: 0.4873
    Epoch 6/12
    88230/88230 [==============================] - 21s 243us/step - loss: 2.5622 - acc: 0.5006 - val_loss: 2.4780 - val_acc: 0.5387
    Epoch 7/12
    88230/88230 [==============================] - 21s 243us/step - loss: 2.3795 - acc: 0.5375 - val_loss: 2.2061 - val_acc: 0.5484
    Epoch 8/12
    88230/88230 [==============================] - 21s 243us/step - loss: 2.4343 - acc: 0.5455 - val_loss: 2.1943 - val_acc: 0.5506
    Epoch 9/12
    88230/88230 [==============================] - 21s 243us/step - loss: 2.2863 - acc: 0.5297 - val_loss: 2.2353 - val_acc: 0.5431
    Epoch 10/12
    88230/88230 [==============================] - 21s 243us/step - loss: 2.1549 - acc: 0.5555 - val_loss: 2.2050 - val_acc: 0.5522
    Epoch 11/12
    88230/88230 [==============================] - 21s 243us/step - loss: 2.1453 - acc: 0.5571 - val_loss: 2.0621 - val_acc: 0.5723
    Epoch 12/12
    88230/88230 [==============================] - 21s 243us/step - loss: 2.0591 - acc: 0.5700 - val_loss: 2.0212 - val_acc: 0.5768





    <keras.callbacks.History at 0x7f9313374b00>



### Embedded Model Scoring on Test Data


```python
test_x = pad(preproc_english_sentences_test, max_french_sequence_length_test)
test_x = test_x.reshape((-1, preproc_french_sentences_train.shape[-2]))

preproc_french_sentences_test = preproc_french_sentences_test.reshape((-1, 
                                                                       preproc_french_sentences_train.shape[-2], 1))

simple_embed_model_score = simple_embed_model_split.evaluate(test_x, preproc_french_sentences_test, verbose=1)

print("Simple Embedding model accuracy on test dataset: {0:.2f}%".format(simple_embed_model_score[1]*100))
```

    26260/26260 [==============================] - 18s 670us/step
    Simple Embedding model accuracy on test dataset: 30.78%



```python
print("Original English sentence: \n", english_sentences_test[1])

print("\nPredicted sentece: \n", logits_to_text(simple_embed_model_split.predict(test_x)[0], french_tokenizer_test))

print("\nOriginal French sentence: \n", french_sentences_test[1])
```

    Original English sentence: 
     california is never hot during autumn , but it is never dry in march .
    
    Predicted sentece: 
     fait est parfois parfois en automne mais il il est est est en <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    Original French sentence: 
     californie est jamais chaud pendant l' automne , mais il est jamais sec en mars .


### Model 3: Bidirectional RNNs 
![RNN](images/bidirectional.png)
One restriction of a RNN is that it can't see the future input, only the past.  This is where bidirectional recurrent neural networks come in.  They are able to see the future data.

### Bidirectional RNN on Training and Test Data


```python
# TODO: Reshape the input
tmp_x = pad(preproc_english_sentences_train, max_french_sequence_length_train)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences_train.shape[-2], 1))

test_x = pad(preproc_english_sentences_test, max_french_sequence_length_test)
test_x = test_x.reshape((-1, preproc_french_sentences_train.shape[-2], 1))

preproc_french_sentences_test = preproc_french_sentences_test.reshape((-1, 
                                preproc_french_sentences_train.shape[-2], 1))
```


```python
def bd_model_split(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a bidirectional RNN model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement
    
    model = Sequential()
    #model.add(Embedding(input_dim=english_vocab_size, output_dim=output_sequence_length, input_shape=input_shape[1:][0]))
    model.add(Bidirectional(GRU(english_vocab_size, return_sequences=True), input_shape=input_shape[1:]))
    #model.add(Dense(french_vocab_size, activation='relu'))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))

    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(.001), metrics=['acc'])        
    model.summary()
    return model

    
tests.test_bd_model(bd_model_split)

```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    bidirectional_1 (Bidirection (None, 21, 398)           239994    
    _________________________________________________________________
    time_distributed_1 (TimeDist (None, 21, 344)           137256    
    =================================================================
    Total params: 377,250
    Trainable params: 377,250
    Non-trainable params: 0
    _________________________________________________________________



```python
# TODO: Train the neural network
bd_model_split = bd_model_split(
    tmp_x.shape,
    max_french_sequence_length_train,
    english_vocab_size,
    french_vocab_size)

# TODO: Print prediction(s)
bd_model_split.fit(tmp_x, preproc_french_sentences_train, batch_size=1024, epochs=10, validation_split=0.2)

print(logits_to_text(bd_model_split.predict(tmp_x[:1])[0], french_tokenizer_train))
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    bidirectional_2 (Bidirection (None, 21, 398)           239994    
    _________________________________________________________________
    time_distributed_2 (TimeDist (None, 21, 344)           137256    
    =================================================================
    Total params: 377,250
    Trainable params: 377,250
    Non-trainable params: 0
    _________________________________________________________________
    Train on 88230 samples, validate on 22058 samples
    Epoch 1/10
    88230/88230 [==============================] - 15s 171us/step - loss: 2.4570 - acc: 0.5244 - val_loss: 1.6496 - val_acc: 0.5883
    Epoch 2/10
    88230/88230 [==============================] - 12s 141us/step - loss: 1.4906 - acc: 0.6127 - val_loss: 1.3772 - val_acc: 0.6268
    Epoch 3/10
    88230/88230 [==============================] - 12s 142us/step - loss: 1.3110 - acc: 0.6395 - val_loss: 1.2549 - val_acc: 0.6492
    Epoch 4/10
    88230/88230 [==============================] - 13s 142us/step - loss: 1.2094 - acc: 0.6572 - val_loss: 1.1721 - val_acc: 0.6619
    Epoch 5/10
    88230/88230 [==============================] - 12s 141us/step - loss: 1.1360 - acc: 0.6709 - val_loss: 1.1069 - val_acc: 0.6778
    Epoch 6/10
    88230/88230 [==============================] - 13s 142us/step - loss: 1.0777 - acc: 0.6822 - val_loss: 1.0522 - val_acc: 0.6872
    Epoch 7/10
    88230/88230 [==============================] - 13s 142us/step - loss: 1.0280 - acc: 0.6909 - val_loss: 1.0178 - val_acc: 0.6899
    Epoch 8/10
    88230/88230 [==============================] - 12s 141us/step - loss: 0.9876 - acc: 0.6991 - val_loss: 0.9708 - val_acc: 0.7044
    Epoch 9/10
    88230/88230 [==============================] - 13s 142us/step - loss: 0.9501 - acc: 0.7075 - val_loss: 0.9366 - val_acc: 0.7093
    Epoch 10/10
    88230/88230 [==============================] - 12s 141us/step - loss: 0.9159 - acc: 0.7161 - val_loss: 0.8988 - val_acc: 0.7216
    l' est est généralement en en et et il est agréable agréable en en <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>


### Bidirectional Model Scoring


```python
bd_model_split_score = bd_model_split.evaluate(test_x, preproc_french_sentences_test, verbose=1)

print("Bidirectional model accuracy on test dataset: {0:.2f}%".format(bd_model_split_score[1]*100))
```

    26260/26260 [==============================] - 14s 532us/step
    Bidirectional model accuracy on test dataset: 29.21%



```python
print("Original English sentence: \n", english_sentences_test[1])

print("\nPredicted sentece: \n", logits_to_text(bd_model_split.predict(test_x)[0], french_tokenizer_test))

print("\nOriginal French sentence: \n", french_sentences_test[1])
```

    Original English sentence: 
     california is never hot during autumn , but it is never dry in march .
    
    Predicted sentece: 
     inde est parfois son pamplemousse l' votre il est parfois parfois au octobre été été mais mais <PAD> est parfois <PAD>
    
    Original French sentence: 
     californie est jamais chaud pendant l' automne , mais il est jamais sec en mars .


### Model 4: Encoder-Decoder
This model is made up of an encoder and decoder. The encoder creates a matrix representation of the sentence.  The decoder takes this matrix as input and predicts the translation as output.

### Encoder-Decoder on Training and Test Sets


```python
def encdec_model_split(input_shape, output_sequence_length, english_vocab_size_train, french_vocab_size_train):
    """
    Build and train an encoder-decoder model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """

    model = Sequential()
    model.add(GRU(english_vocab_size, return_sequences=False, input_shape=input_shape[1:]))
    model.add(Dense(french_vocab_size, activation='sigmoid'))
    model.add(RepeatVector(output_sequence_length))
    model.add(GRU(english_vocab_size, return_sequences=True))
    model.add(TimeDistributed(Dense(french_vocab_size)))
    model.add(Activation('softmax'))

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(10e-3),
                  metrics=['accuracy'])
    model.summary()
    return model
    
    return None
tests.test_encdec_model(encdec_model_split)

# TODO: Reshape the input
tmp_x = pad(preproc_english_sentences_train, max_french_sequence_length_train)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences_train.shape[-2], 1))
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    gru_6 (GRU)                  (None, 199)               119997    
    _________________________________________________________________
    dense_6 (Dense)              (None, 344)               68800     
    _________________________________________________________________
    repeat_vector_1 (RepeatVecto (None, 21, 344)           0         
    _________________________________________________________________
    gru_7 (GRU)                  (None, 21, 199)           324768    
    _________________________________________________________________
    time_distributed_6 (TimeDist (None, 21, 344)           68800     
    _________________________________________________________________
    activation_1 (Activation)    (None, 21, 344)           0         
    =================================================================
    Total params: 582,365
    Trainable params: 582,365
    Non-trainable params: 0
    _________________________________________________________________



```python
# Build the model
encdec_rnn_model_split = encdec_model_split(
    tmp_x.shape,
    max_french_sequence_length_train,
    english_vocab_size,
    french_vocab_size)

# TODO: Train the neural network
encdec_rnn_model_split.fit(tmp_x, preproc_french_sentences_train, batch_size=1024, epochs=12, validation_split=0.2)

# TODO: Print prediction(s)
print(logits_to_text(encdec_rnn_model_split.predict(tmp_x[:1])[0], french_tokenizer_train))

```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    gru_8 (GRU)                  (None, 199)               119997    
    _________________________________________________________________
    dense_8 (Dense)              (None, 344)               68800     
    _________________________________________________________________
    repeat_vector_2 (RepeatVecto (None, 21, 344)           0         
    _________________________________________________________________
    gru_9 (GRU)                  (None, 21, 199)           324768    
    _________________________________________________________________
    time_distributed_7 (TimeDist (None, 21, 344)           68800     
    _________________________________________________________________
    activation_2 (Activation)    (None, 21, 344)           0         
    =================================================================
    Total params: 582,365
    Trainable params: 582,365
    Non-trainable params: 0
    _________________________________________________________________
    Train on 88230 samples, validate on 22058 samples
    Epoch 1/12
    88230/88230 [==============================] - 16s 177us/step - loss: 2.8468 - acc: 0.4293 - val_loss: 2.3964 - val_acc: 0.4642
    Epoch 2/12
    88230/88230 [==============================] - 15s 168us/step - loss: 2.2523 - acc: 0.4686 - val_loss: 2.1911 - val_acc: 0.4767
    Epoch 3/12
    88230/88230 [==============================] - 15s 168us/step - loss: 2.1611 - acc: 0.4775 - val_loss: 2.0767 - val_acc: 0.4813
    Epoch 4/12
    88230/88230 [==============================] - 15s 168us/step - loss: 1.9720 - acc: 0.5028 - val_loss: 1.8339 - val_acc: 0.5275
    Epoch 5/12
    88230/88230 [==============================] - 15s 168us/step - loss: 1.7962 - acc: 0.5408 - val_loss: 1.8053 - val_acc: 0.5378
    Epoch 6/12
    88230/88230 [==============================] - 15s 168us/step - loss: 1.6475 - acc: 0.5689 - val_loss: 1.5586 - val_acc: 0.5834
    Epoch 7/12
    88230/88230 [==============================] - 15s 169us/step - loss: 1.5382 - acc: 0.5841 - val_loss: 1.5090 - val_acc: 0.5858
    Epoch 8/12
    88230/88230 [==============================] - 15s 168us/step - loss: 1.4618 - acc: 0.5966 - val_loss: 1.5119 - val_acc: 0.5933
    Epoch 9/12
    88230/88230 [==============================] - 15s 168us/step - loss: 1.4090 - acc: 0.6071 - val_loss: 1.4252 - val_acc: 0.6033
    Epoch 10/12
    88230/88230 [==============================] - 15s 169us/step - loss: 1.3949 - acc: 0.6089 - val_loss: 1.3562 - val_acc: 0.6193
    Epoch 11/12
    88230/88230 [==============================] - 15s 168us/step - loss: 1.4013 - acc: 0.6072 - val_loss: 1.3413 - val_acc: 0.6264
    Epoch 12/12
    88230/88230 [==============================] - 15s 168us/step - loss: 1.4370 - acc: 0.6046 - val_loss: 1.3341 - val_acc: 0.6268
    la est est jamais en en mais il est est est en en <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>


### Encoder-Decoder Model Score


```python
test_x = pad(preproc_english_sentences_test, max_french_sequence_length_test)
test_x = test_x.reshape((-1, preproc_french_sentences_train.shape[-2], 1))

preproc_french_sentences_test = preproc_french_sentences_test.reshape((-1, 
                                preproc_french_sentences_train.shape[-2], 1))

encdec_rnn_model_split_score = encdec_rnn_model_split.evaluate(test_x, preproc_french_sentences_test, verbose=1)

print("Encoder-Decoder model accuracy on test dataset: {0:.2f}%".format(bd_model_split_score[1]*100))
```

    26260/26260 [==============================] - 14s 544us/step
    Encoder-Decoder model accuracy on test dataset: 29.21%



```python
print("Original English sentence: \n", english_sentences_test[1])

print("\nPredicted sentece: \n", logits_to_text(bd_model_split.predict(test_x)[0], french_tokenizer_test))

print("\nOriginal French sentence: \n", french_sentences_test[1])
```

    Original English sentence: 
     california is never hot during autumn , but it is never dry in march .
    
    Predicted sentece: 
     inde est parfois son pamplemousse l' votre il est parfois parfois au octobre été été mais mais <PAD> est parfois <PAD>
    
    Original French sentence: 
     californie est jamais chaud pendant l' automne , mais il est jamais sec en mars .


### Model 5: Custom 
Using everything I learned from the previous models to create a model that incorporates embedding and a bidirectional rnn into one model.

### Final Model on Training and Test Data


```python

def model_final_split(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    
    model = Sequential()
    model.add(Embedding(300, output_dim=output_sequence_length,
                        input_length=input_shape[1:][0]))
    model.add(LSTM(512, return_sequences=True,input_shape=input_shape))
    model.add(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))          
    model.add(Flatten())
    model.add(RepeatVector(output_sequence_length))
    model.add(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)) 
    model.add(Bidirectional(GRU(512, return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))

    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(lr=0.001),
                  metrics=['acc'])
    model.summary()
    return model


tests.test_model_final(model_final_split)

#Reshape the input
tmp_x = pad(preproc_english_sentences_train, max_french_sequence_length_train)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences_train.shape[-2]))
tmp_x.shape

print('Final Model Loaded')
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_7 (Embedding)      (None, 15, 21)            6300      
    _________________________________________________________________
    lstm_31 (LSTM)               (None, 15, 512)           1093632   
    _________________________________________________________________
    lstm_32 (LSTM)               (None, 15, 512)           2099200   
    _________________________________________________________________
    lstm_33 (LSTM)               (None, 15, 512)           2099200   
    _________________________________________________________________
    flatten_7 (Flatten)          (None, 7680)              0         
    _________________________________________________________________
    repeat_vector_7 (RepeatVecto (None, 21, 7680)          0         
    _________________________________________________________________
    lstm_34 (LSTM)               (None, 21, 512)           16779264  
    _________________________________________________________________
    lstm_35 (LSTM)               (None, 21, 512)           2099200   
    _________________________________________________________________
    bidirectional_7 (Bidirection (None, 21, 1024)          3148800   
    _________________________________________________________________
    time_distributed_6 (TimeDist (None, 21, 344)           352600    
    =================================================================
    Total params: 27,678,196
    Trainable params: 27,678,196
    Non-trainable params: 0
    _________________________________________________________________
    Final Model Loaded



```python
# Build the model
model_final_split = model_final_split(
    tmp_x.shape,
    max_french_sequence_length_train,
    english_vocab_size,
    french_vocab_size)

# TODO: Train the neural network
model_final_split.fit(tmp_x, preproc_french_sentences_train, batch_size=1024, epochs=20, validation_split=0.2)

# TODO: Print prediction(s)
print(logits_to_text(model_final_split.predict(tmp_x[:1])[0], french_tokenizer_train))

```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_8 (Embedding)      (None, 21, 21)            6300      
    _________________________________________________________________
    lstm_36 (LSTM)               (None, 21, 512)           1093632   
    _________________________________________________________________
    lstm_37 (LSTM)               (None, 21, 512)           2099200   
    _________________________________________________________________
    lstm_38 (LSTM)               (None, 21, 512)           2099200   
    _________________________________________________________________
    flatten_8 (Flatten)          (None, 10752)             0         
    _________________________________________________________________
    repeat_vector_8 (RepeatVecto (None, 21, 10752)         0         
    _________________________________________________________________
    lstm_39 (LSTM)               (None, 21, 512)           23070720  
    _________________________________________________________________
    lstm_40 (LSTM)               (None, 21, 512)           2099200   
    _________________________________________________________________
    bidirectional_8 (Bidirection (None, 21, 1024)          3148800   
    _________________________________________________________________
    time_distributed_7 (TimeDist (None, 21, 344)           352600    
    =================================================================
    Total params: 33,969,652
    Trainable params: 33,969,652
    Non-trainable params: 0
    _________________________________________________________________
    Train on 88230 samples, validate on 22058 samples
    Epoch 1/20
    88230/88230 [==============================] - 341s 4ms/step - loss: 2.6723 - acc: 0.4700 - val_loss: 1.9310 - val_acc: 0.5326
    Epoch 2/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 1.7221 - acc: 0.5607 - val_loss: 1.5563 - val_acc: 0.5741
    Epoch 3/20
    88230/88230 [==============================] - 335s 4ms/step - loss: 1.4026 - acc: 0.6141 - val_loss: 1.2749 - val_acc: 0.6397
    Epoch 4/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 1.2200 - acc: 0.6553 - val_loss: 1.2185 - val_acc: 0.6504
    Epoch 5/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 1.0975 - acc: 0.6820 - val_loss: 1.0463 - val_acc: 0.6919
    Epoch 6/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 1.0088 - acc: 0.6989 - val_loss: 0.9404 - val_acc: 0.7139
    Epoch 7/20
    88230/88230 [==============================] - 335s 4ms/step - loss: 0.9160 - acc: 0.7171 - val_loss: 0.8918 - val_acc: 0.7222
    Epoch 8/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 0.8607 - acc: 0.7280 - val_loss: 0.8212 - val_acc: 0.7353
    Epoch 9/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 0.7878 - acc: 0.7452 - val_loss: 0.7581 - val_acc: 0.7526
    Epoch 10/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 0.7255 - acc: 0.7611 - val_loss: 0.7169 - val_acc: 0.7629
    Epoch 11/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 0.6825 - acc: 0.7732 - val_loss: 0.6724 - val_acc: 0.7771
    Epoch 12/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 0.6477 - acc: 0.7836 - val_loss: 0.6060 - val_acc: 0.7962
    Epoch 13/20
    88230/88230 [==============================] - 335s 4ms/step - loss: 0.5928 - acc: 0.8001 - val_loss: 0.5950 - val_acc: 0.7980
    Epoch 14/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 0.5676 - acc: 0.8076 - val_loss: 0.5331 - val_acc: 0.8176
    Epoch 15/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 0.5391 - acc: 0.8166 - val_loss: 0.5260 - val_acc: 0.8216
    Epoch 16/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 0.5035 - acc: 0.8285 - val_loss: 0.4956 - val_acc: 0.8324
    Epoch 17/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 0.4661 - acc: 0.8408 - val_loss: 0.4330 - val_acc: 0.8525
    Epoch 18/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 0.4265 - acc: 0.8533 - val_loss: 0.4081 - val_acc: 0.8592
    Epoch 19/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 0.3976 - acc: 0.8622 - val_loss: 0.3839 - val_acc: 0.8666
    Epoch 20/20
    88230/88230 [==============================] - 334s 4ms/step - loss: 0.3676 - acc: 0.8714 - val_loss: 0.3487 - val_acc: 0.8777
    l' inde est parfois gel habituellement en janvier mais il est jamais agréable à l' automne <PAD> <PAD> <PAD> <PAD> <PAD>


### Final Model Scoring


```python
test_x = pad(preproc_english_sentences_test, max_french_sequence_length_test)
test_x = test_x.reshape((-1, preproc_french_sentences_train.shape[-2]))

preproc_french_sentences_test = preproc_french_sentences_test.reshape((-1, 
                                preproc_french_sentences_train.shape[-2], 1))

model_final_split_score = model_final_split.evaluate(test_x, preproc_french_sentences_test, verbose=1)

print("Final model accuracy on test dataset: {0:.2f}%".format(model_final_split_score[1]*100))
```

    26260/26260 [==============================] - 113s 4ms/step
    Final model accuracy on test dataset: 28.78%



```python
print("Original English sentence: \n", english_sentences_test[1])

print("\nPredicted sentece: \n", logits_to_text(model_final_split.predict(test_x)[0], french_tokenizer_test))

print("\nOriginal French sentence: \n", french_sentences_test[1])
```

    Original English sentence: 
     california is never hot during autumn , but it is never dry in march .
    
    Predicted sentece: 
     inde est parfois pluvieux pamplemousse l' mois mais il belle jamais jamais chaud à l' mois <PAD> <PAD> <PAD> <PAD> <PAD>
    
    Original French sentence: 
     californie est jamais chaud pendant l' automne , mais il est jamais sec en mars .


### Final Model w/Less Layers & Nodes


```python
def model_final_3LSTM_split(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    
    model = Sequential()
    model.add(Embedding(300, output_dim=output_sequence_length,
                        input_length=input_shape[1:][0]))
    model.add(LSTM(128, return_sequences=True,input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))          
    model.add(Flatten())
    model.add(RepeatVector(output_sequence_length))
    model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(Bidirectional(GRU(512, return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))

    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(lr=0.001),
                  metrics=['acc'])
    model.summary()
    return model


tests.test_model_final(model_final_3LSTM_split)

#Reshape the input
tmp_x = pad(preproc_english_sentences_train, max_french_sequence_length_train)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences_train.shape[-2]))
tmp_x.shape

print('Final Model Loaded')
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_9 (Embedding)      (None, 15, 21)            6300      
    _________________________________________________________________
    lstm_41 (LSTM)               (None, 15, 128)           76800     
    _________________________________________________________________
    lstm_42 (LSTM)               (None, 15, 128)           131584    
    _________________________________________________________________
    flatten_9 (Flatten)          (None, 1920)              0         
    _________________________________________________________________
    repeat_vector_9 (RepeatVecto (None, 21, 1920)          0         
    _________________________________________________________________
    lstm_43 (LSTM)               (None, 21, 128)           1049088   
    _________________________________________________________________
    bidirectional_9 (Bidirection (None, 21, 1024)          1969152   
    _________________________________________________________________
    time_distributed_8 (TimeDist (None, 21, 344)           352600    
    =================================================================
    Total params: 3,585,524
    Trainable params: 3,585,524
    Non-trainable params: 0
    _________________________________________________________________
    Final Model Loaded



```python
# Build the model
model_final_3LSTM_split = model_final_3LSTM_split(
    tmp_x.shape,
    max_french_sequence_length_train,
    english_vocab_size,
    french_vocab_size)

# TODO: Train the neural network
model_final_3LSTM_split.fit(tmp_x, preproc_french_sentences_train, batch_size=1024, epochs=20, validation_split=0.2)

# TODO: Print prediction(s)
print(logits_to_text(model_final_3LSTM_split.predict(tmp_x[:1])[0], french_tokenizer_train))

```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_10 (Embedding)     (None, 21, 21)            6300      
    _________________________________________________________________
    lstm_44 (LSTM)               (None, 21, 128)           76800     
    _________________________________________________________________
    lstm_45 (LSTM)               (None, 21, 128)           131584    
    _________________________________________________________________
    flatten_10 (Flatten)         (None, 2688)              0         
    _________________________________________________________________
    repeat_vector_10 (RepeatVect (None, 21, 2688)          0         
    _________________________________________________________________
    lstm_46 (LSTM)               (None, 21, 128)           1442304   
    _________________________________________________________________
    bidirectional_10 (Bidirectio (None, 21, 1024)          1969152   
    _________________________________________________________________
    time_distributed_9 (TimeDist (None, 21, 344)           352600    
    =================================================================
    Total params: 3,978,740
    Trainable params: 3,978,740
    Non-trainable params: 0
    _________________________________________________________________
    Train on 88230 samples, validate on 22058 samples
    Epoch 1/20
    88230/88230 [==============================] - 65s 733us/step - loss: 2.7760 - acc: 0.4621 - val_loss: 2.2332 - val_acc: 0.4997
    Epoch 2/20
    88230/88230 [==============================] - 61s 696us/step - loss: 1.9244 - acc: 0.5278 - val_loss: 1.6307 - val_acc: 0.5666
    Epoch 3/20
    88230/88230 [==============================] - 61s 696us/step - loss: 1.4879 - acc: 0.5942 - val_loss: 1.3895 - val_acc: 0.6206
    Epoch 4/20
    88230/88230 [==============================] - 61s 695us/step - loss: 1.3106 - acc: 0.6338 - val_loss: 1.2331 - val_acc: 0.6499
    Epoch 5/20
    88230/88230 [==============================] - 61s 695us/step - loss: 1.1876 - acc: 0.6615 - val_loss: 1.1423 - val_acc: 0.6733
    Epoch 6/20
    88230/88230 [==============================] - 61s 695us/step - loss: 1.1203 - acc: 0.6753 - val_loss: 1.0846 - val_acc: 0.6832
    Epoch 7/20
    88230/88230 [==============================] - 61s 695us/step - loss: 1.0531 - acc: 0.6898 - val_loss: 1.0532 - val_acc: 0.6893
    Epoch 8/20
    88230/88230 [==============================] - 61s 696us/step - loss: 1.0020 - acc: 0.7005 - val_loss: 0.9684 - val_acc: 0.7073
    Epoch 9/20
    88230/88230 [==============================] - 61s 693us/step - loss: 0.9532 - acc: 0.7108 - val_loss: 0.9295 - val_acc: 0.7145
    Epoch 10/20
    88230/88230 [==============================] - 61s 695us/step - loss: 0.9113 - acc: 0.7188 - val_loss: 0.8838 - val_acc: 0.7208
    Epoch 11/20
    88230/88230 [==============================] - 61s 696us/step - loss: 0.8621 - acc: 0.7302 - val_loss: 0.8258 - val_acc: 0.7392
    Epoch 12/20
    88230/88230 [==============================] - 61s 696us/step - loss: 0.8188 - acc: 0.7427 - val_loss: 0.7902 - val_acc: 0.7513
    Epoch 13/20
    88230/88230 [==============================] - 61s 694us/step - loss: 0.7760 - acc: 0.7555 - val_loss: 0.7469 - val_acc: 0.7641
    Epoch 14/20
    88230/88230 [==============================] - 61s 695us/step - loss: 0.7344 - acc: 0.7671 - val_loss: 0.7071 - val_acc: 0.7734
    Epoch 15/20
    88230/88230 [==============================] - 61s 693us/step - loss: 0.6908 - acc: 0.7790 - val_loss: 0.6964 - val_acc: 0.7769
    Epoch 16/20
    88230/88230 [==============================] - 61s 695us/step - loss: 0.6550 - acc: 0.7879 - val_loss: 0.6207 - val_acc: 0.7972
    Epoch 17/20
    88230/88230 [==============================] - 61s 695us/step - loss: 0.6151 - acc: 0.7986 - val_loss: 0.5782 - val_acc: 0.8106
    Epoch 18/20
    88230/88230 [==============================] - 61s 696us/step - loss: 0.5826 - acc: 0.8070 - val_loss: 0.5457 - val_acc: 0.8189
    Epoch 19/20
    88230/88230 [==============================] - 61s 695us/step - loss: 0.5467 - acc: 0.8171 - val_loss: 0.5403 - val_acc: 0.8194
    Epoch 20/20
    88230/88230 [==============================] - 61s 696us/step - loss: 0.5162 - acc: 0.8260 - val_loss: 0.4951 - val_acc: 0.8309
    l' inde est le le habituellement en juillet mais il est jamais agréable à l' automne <PAD> <PAD> <PAD> <PAD> <PAD>



```python
test_x = pad(preproc_english_sentences_test, max_french_sequence_length_test)
test_x = test_x.reshape((-1, preproc_french_sentences_train.shape[-2]))

preproc_french_sentences_test = preproc_french_sentences_test.reshape((-1, 
                                preproc_french_sentences_train.shape[-2], 1))

model_final_3LSTM_score = model_final_3LSTM_split.evaluate(test_x, preproc_french_sentences_test, verbose=1)

print("Final model w/3 LSTM layers' accuracy on test dataset: {0:.2f}%".format(model_final_3LSTM_score[1]*100))
```

    26260/26260 [==============================] - 48s 2ms/step
    Final model w/3 LSTM layers' accuracy on test dataset: 27.42%



```python
print("Original English sentence: \n", english_sentences_test[1])

print("\nPredicted sentece: \n", logits_to_text(model_final_3LSTM_split.predict(test_x)[0], french_tokenizer_test))

print("\nOriginal French sentence: \n", french_sentences_test[1])
```

    Original English sentence: 
     california is never hot during autumn , but it is never dry in march .
    
    Predicted sentece: 
     inde est parfois son pamplemousse l' printemps mais il est jamais chaud chaud <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
    
    Original French sentence: 
     californie est jamais chaud pendant l' automne , mais il est jamais sec en mars .

