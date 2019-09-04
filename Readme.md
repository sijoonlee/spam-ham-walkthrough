#  Spam-Ham Classification Using LSTM model on PyTorch: Step-by-step walkthrough

This is how to train LSTM model in PyTorch and use it to predict Spam or Ham.  
You can see [the medium version](https://medium.com/@shijoonlee/spam-ham-classification-using-lstm-in-pytorch-950daec94a7c) of this readme.md

## I. Enron Spam Datasets 
Researchers - V. Metsis, I. Androutsopoulos and G. Paliouras - classified over 30,000 emails in the Enron corpus as Spam/Ham datasets and have had them open to the public

1. Go to [the website](http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html)
2. Find `Enron-Spam in pre-processed form` in the site
3. Download Enron1, Enron2, Enron3, Enron4, Enron5 and Enron6
4. Extract each tar.gz file
5. Directories - enron1, enron2, ... , enron6 should be under the same parent directory

## II. Processing data
Data will be  
1) loaded from files  
2) used to build vocabulary dictionary  
3) tokenized and vetorized  

Let's dive into each step

### II-1. Load data from files

How the data is loaded from files is out of focus in this document.  
Therefore, I will briefly introduce [the code]() that I wrote.  

At first, spam and ham sets will be loaded into `ham` and `spam`  
Secondly, `ham` and `spam` will be merged into `data`  
Thirdly, labels will be generated for spam and ham, 1 and 0, respectively  
Lastly, it returns data and labels  

https://gist.github.com/sijoonlee/eb0fe90c447bbc4e80b8e3619d7fc76e.js

The loaded data consists of 3000 of hams and 3000 of spams - In total, 6000  
If you set `max = 0`, you can get all the data from files.  
But for this tutorial, 6000 sets are enough

### II-2.  Build Vocab dictionary

Vocabulary dictionary has keys and values in it: Keys are words and values are numbers(integer).  
e.g. {'the': 2, 'to': 3}

How to set the numbers in the dictionary?  

Please imagine that we've got a list of words from 6,000 datasets.  
Common words like "the", "to" and "and" are more likely to be present in the lists.
We will count the number of occurrences and order the words by their counts.

https://gist.github.com/sijoonlee/dbd4929b4ffa2c2782057f85a3276740.js

### II-3. Tokenize & Vectorize data
`Tokenization` means here the conversion from datasets into lists of words.
For example, let say we have data like below
```
"operations is digging out 2000 feet of pipe to begin the hydro test"
```
Tokenization will produce a list of words like below
```
['operations', 'is', 'digging', ...
```
`Vectorization` means here the conversion from words into integer numbers using the vocab dictionary built in II-2
```
[424, 11, 14683, ...
```  
See the example codes
https://gist.github.com/sijoonlee/7c46ca2d948d9cf2954d0ff8d91297c3.js

Below is the code for our spam-ham datasets
https://gist.github.com/sijoonlee/b26ed463febb70aa3749d5b8d63b0145.js


## III. Build Data loaders
So far, data is processed and prepared as vectorized form.  
Now, it is turn to build data loaders that will feed the batches of datasets into our model.  

To do so,  
1) A custom data loader class was built
2) 3 data loaders were instantiated : for train, validation, and test

### III-1. Custom Data Loader
`Sequence` here means a vectorized list of words in an email.  
As we prepared 6,000 e-mails, we have 6,000 sequences.

As sequences have different lengths, it is required to pass the length of each sequence into our model not to train our model on dummy numbers ( 0s for padding ).

Consequently, we need custom data loaders that return lengths of each sequence along with sequences and labels.

Plus, The data loader should sort the batch by each sequence's length and returns the longest one first in the batch to use torch's `pack_padded_sequence()` (you will see this function in model code)

I built the iterable data loader class using torch's sampler.
https://gist.github.com/sijoonlee/0e13ffe5888ad50469c5ab17a5b336ea

### III-2. Instantiate 3 Data Loaders
Model will be trained on **train** datasets, be validated by **validation** dataset, and finally be tested on **test** datasets

https://gist.github.com/sijoonlee/8df9e1297f686468b304611c96081477


## IV. Structure the model

The model consists of
1) Embedding
2) Pack the sequences (get rid of paddings)
3) LSTM
4) Unpack the sequences (recover paddings)
4) Fully Connected Layer
5) Sigmoid Activation

https://gist.github.com/sijoonlee/66dcebb174b6277ff4ebff39c48910e4.js

### IV-1. Embedding
According to PyTorch.org's documentation, "word embeddings are a representation of the *semantics* of a word"

> To know what the `Word Embeddings` is, I would recommend you to read [PyTorch Documentation](*https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)

### IV-2. Use of pack_padded_sequence()
Please recall that we added padding(0)s to sequences.
Since sequences have different lengths, it is required to add paddings into shorter sequences to match the dimension in tensor.
The problem is that model should not be trained on padding values.
pack_padded_sequence() will get rid of paddings in the batch of data and re-organized it

For example,
https://gist.github.com/sijoonlee/fbe7c1ac76b133b4a7d7c25e43a730b9.js

> To understand more about `pack_padded_sequence()`,
I would recommend you to read [layog's stack overflow post](https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch/49473068#49473068) and [HarshTrivedi's tutorial](https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial)

### IV-3. LSTM

It stands for "Long short-term memory", a kind of RNN architecture.
Note that, If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero according to [PyTorch documentation](https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html)

> For `LSTM`, I would recommend you to read [colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)


## V. Train, Validate and Test

V-1. Train and Validate
https://gist.github.com/sijoonlee/1e25e391bd1b56e0bffe74c0ddefa89c.js

V-2. Test
https://gist.github.com/sijoonlee/52204f2d0513010bfbeed9ade350bf11.js

## VI. Predict with a real case
This is a part of the advertisement of English lesson I've got recently
```
Have you been really busy this week?
Then you'll definitely want to make time for this lesson.
Have a wonderful week, learn something new, and practice some English!
```
Let's put this into model and see if the result is "Spam"
https://gist.github.com/sijoonlee/97c37690a8c98f0b49f7e588b3c70b44.js




Thank you for your reading!  
I never have expected myself writing a guide since I still see myself as a beginner in deep learning. If you find something wrong, please email me or leave your comments, it would be greatly appreciated.

[Email](shijoonlee@gmail.com)
[Github](https://github.com/sijoonlee)
