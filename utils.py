import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

#nltk.download('punkt')

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(token_sentence, vocab):
    token_sentence = [stem(token) for token in token_sentence]
    bag = np.zeros_like(vocab, dtype=np.float32)
    for i, word in enumerate(vocab):
        if word in token_sentence:
            bag[i] = 1

    return bag
