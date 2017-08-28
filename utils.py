import pickle
import os
import sqlite3
import numpy as np


def exists(path, filename, vocab_name):
    return os.path.isfile(os.path.join(path, filename) + "." + vocab_name + ".pickle")


def load_glove(path, filename, vocab_name, vocab=None):
    """
    :arg path: Path to directory holding wordvec files
    :arg filename: Name of .txt file holding GloVe vectors
    :arg vocab_name: Name of the vocabulary to store wordvecs from
    :arg vocab: A Python set of strings to be used as the vocabulary to store wordvecs from

    Returns a tuple (words, wordvecs) of the form:

    words: a dictionary mapping from string => int
    wordvecs: a list of length vocab_size, with entries that are numpy arrays of length word_dims

    wordvecs[words["word"]] is the word vector for "word"
    """
    if os.path.isfile(os.path.join(path, filename) + "." + vocab_name + ".pickle"):
        return pickle.load(open(os.path.join(path, filename) + "." + vocab_name + ".pickle", "rb"))

    else:
        if vocab is None:
            raise RuntimeError("Pickled wordvec file does not exist yet. Must supply vocab.")
        wordvecs = []
        words = {}
        with open(os.path.join(path, filename), "r", encoding="UTF-8") as f:
            lines = f.readlines()
            word_idx = 0
            for i in range(len(lines)):
                if i % 10000 == 0: print("glove: %d%%" % (i / len(lines) * 100))
                line = lines[i]
                l = line.split(' ')
                if l[0].lower() in vocab and l[0].lower() not in words.keys():
                    wordvecs.append(np.array([float(i) for i in l[1:]]))
                    words[l[0].lower()] = word_idx
                    word_idx += 1
        print("glove: pickling")
        wordvecs = np.array(wordvecs)
        pickle.dump((words, wordvecs), open(os.path.join(path, filename) + "." + vocab_name + ".pickle", "wb"), protocol=4)
        return words, wordvecs


def load_word2vec(path, filename):
    if os.path.isfile(os.path.join(path, filename) + ".pickle"):
        return pickle.load(open(os.path.join(path, filename) + ".pickle", "rb"))
    else:
        wordvecs = {}
        words = []
        with open(os.path.join(path, filename), "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch.decode('cp437'))
                wordvecs[word.lower()] = np.fromstring(f.read(binary_len), dtype='float32')
                words.append(word.lower())
                if (line % 10000 == 0):
                    print("word2vec: %d / %d" % (line, vocab_size))
        print("word2vec: pickling...")
        pickle.dump((wordvecs, words), open(os.path.join(path, filename) + ".pickle", "wb"), protocol=4)
        return wordvecs, words
