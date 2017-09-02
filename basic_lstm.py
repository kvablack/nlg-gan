import tensorflow as tf
import numpy as np
import pickle
from plot import Plotter
from nltk.corpus import gutenberg
from basic_lstm.model import NlpGan
import utils

SAVE_NAME = 'ADAM'

LEARNING_RATE = 0.000004
DIM_STATE = 500
WORD_DIM = 300
SEQUENCE_LENGTH = 50
BATCH_SIZE = 100
INPUT_DROPOUT = 0.65


def nearest_neighbor(words, wordvecs_norm, wordvec_norm):
    similarities = np.dot(wordvecs_norm, wordvec_norm.reshape(-1)).reshape(-1)
    index = np.argmax(similarities)
    return reverse_lookup(words, index)


def nearest_neighbors(words, wordvecs_norm, wordvec_norm, n):
    similarities = np.dot(wordvecs_norm, wordvec_norm.reshape(-1)).reshape(-1)
    indices = np.argpartition(similarities, -1 * n)[-1 * n:]

    indices = indices[np.argsort(similarities[indices])[::-1]]
    similarities = np.sort(similarities[indices])[::-1]

    return [(reverse_lookup(words, i), s) for i, s in zip(indices, similarities)]


def reverse_lookup(dd, val):
    return next(key for key, value in dd.items() if value == val)


def clean_word(word):
    return word.lower().replace('_', '').replace('"', '').replace("'", '')


def pad_or_truncate(sentence):
    if len(sentence) < SEQUENCE_LENGTH:
        return sentence + [np.zeros(WORD_DIM)] * (SEQUENCE_LENGTH - len(sentence))
    else:
        return sentence[:SEQUENCE_LENGTH]


#def rand_swap(sentence):
#    i1, i2 = np.random.randint(0, len(sentence), 2)
#    sentence_fake = list(sentence)
#    sentence_fake[i1], sentence_fake[i2] = sentence_fake[i2], sentence_fake[i1]
#    return sentence_fake

def rand_swap(sentence):
    fake = list(sentence)
    np.random.shuffle(fake)
    return fake


def main():
    print("Loading wordvecs...")
    if utils.exists("glove", "glove.840B.300d.txt", "gutenberg"):
        words, wordvecs = utils.load_glove("glove", "glove.840B.300d.txt", "gutenberg")
    else:
        words, wordvecs = utils.load_glove("glove", "glove.840B.300d.txt", "gutenberg",
                                           set(map(clean_word, gutenberg.words())))

    print("Loading corpus...")
    # Convert corpus into wordvecs, replacing any words not in vocab with zero vector
    sentences = [[wordvecs[words[clean_word(word)]] if clean_word(word) in words.keys() else np.zeros(WORD_DIM) for word in sentence]
                 for sentence in gutenberg.sents()]

    print("Processing corpus...")
    # Randomly swap words in each sentence
    sentences_fake = list(map(rand_swap, sentences))

    # Pad sentences shorter than SEQUENCE_LENGTH with zero vectors and truncate sentences longer than SEQUENCE_LENGTH
    sentences = list(map(pad_or_truncate, sentences))
    sentences_fake = list(map(pad_or_truncate, sentences_fake))

    np.random.shuffle(sentences)
    np.random.shuffle(sentences_fake)
    # Split up into train/eval sections, 97% train and 3% eval
    s_train, s_eval = sentences[:int(len(sentences) * 0.97)], sentences[int(len(sentences) * 0.97):]
    s_train_fake, s_eval_fake = sentences_fake[:int(len(sentences) * 0.97)], sentences_fake[int(len(sentences) * 0.97):]

    # Truncate to multiple of BATCH_SIZE
    s_train = s_train[:int(len(s_train) / BATCH_SIZE) * BATCH_SIZE]
    s_train_fake = s_train_fake[:int(len(s_train_fake) / BATCH_SIZE) * BATCH_SIZE]

    s_eval_len = len(s_eval)
    # reshape to (SEQUENCE_LENGTH, BATCH_SIZE, WORD_DIM) while preserving sentence order; BATCH_SIZE is entire dataset
    # in this case
    s_eval = np.array(s_eval).reshape(-1, WORD_DIM).reshape(SEQUENCE_LENGTH, -1, WORD_DIM, order='F')
    s_eval_fake = np.array(s_eval_fake).reshape(-1, WORD_DIM).reshape(SEQUENCE_LENGTH, -1, WORD_DIM, order='F')

    s_train_idxs = np.arange(len(s_train))
    s_train_fake_idxs = np.arange(len(s_train))

    print("Generating graph...")
    network = NlpGan(learning_rate=LEARNING_RATE, dim_state=DIM_STATE, dim_in=WORD_DIM, sequence_length=SEQUENCE_LENGTH)

    plotter = Plotter([2, 1], "Loss", "Accuracy")
    plotter.plot(0, 0, 0, 0)
    plotter.plot(0, 1, 1, 0)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(100000):
            print("Epoch %d" % epoch)

            np.random.shuffle(s_train_idxs)
            np.random.shuffle(s_train_fake_idxs)
            for batch in range(int(len(s_train_idxs) / BATCH_SIZE)):
                # select next random batch of sentences
                s_batch_real = [s_train[x] for x in s_train_idxs[batch:batch + BATCH_SIZE]] # shape (BATCH_SIZE, SEQUENCE_LENGTH, WORD_DIM)
                s_batch_fake = [s_train_fake[x] for x in s_train_fake_idxs[batch:batch + BATCH_SIZE]]

                # reshape to (SEQUENCE_LENGTH, BATCH_SIZE, WORD_DIM) while preserving sentence order
                s_batch_real = np.array(s_batch_real).reshape(-1, WORD_DIM).reshape(SEQUENCE_LENGTH, BATCH_SIZE, WORD_DIM, order='F')
                s_batch_fake = np.array(s_batch_fake).reshape(-1, WORD_DIM).reshape(SEQUENCE_LENGTH, BATCH_SIZE, WORD_DIM, order='F')

                # real text
                output_dict_real = sess.run(
                    network.get_fetch_dict('d_loss', 'd_train'),
                    network.get_feed_dict(inputs=s_batch_real, labels=np.ones([BATCH_SIZE, 1]),
                                          input_dropout=INPUT_DROPOUT)
                )

                # fake text
                output_dict_fake = sess.run(
                    network.get_fetch_dict('d_loss', 'd_train'),
                    network.get_feed_dict(inputs=s_batch_fake, labels=np.zeros([BATCH_SIZE, 1]),
                                          input_dropout=INPUT_DROPOUT)
                )

                total_loss = (output_dict_real['d_loss'] + output_dict_fake['d_loss']) / 2.0
                if batch % 10 == 0:
                    print("Finished training batch %d / %d" % (batch, int(len(s_train) / BATCH_SIZE)))
                    print("Total Loss: %f" % total_loss)
                    plotter.plot(epoch + (batch / int(len(s_train) / BATCH_SIZE)), total_loss, 0, 0)

                if batch % 100 == 0:
                    eval_accuracy, eval_loss = eval(network, s_eval, s_eval_fake, s_eval_len, sess)
                    print("Total Accuracy: %f" % eval_accuracy)
                    plotter.plot(epoch + (batch / int(len(s_train) / BATCH_SIZE)), eval_accuracy, 1, 0)
                    plotter.plot(epoch + (batch / int(len(s_train) / BATCH_SIZE)), eval_loss, 0, 1)

            saver.save(sess, './checkpoints/{}.ckpt'.format(SAVE_NAME),
                       global_step=epoch)
            plotter.save(SAVE_NAME)


def eval(network, s_eval, s_eval_fake, s_eval_len, sess):
    print("Evaluating accuracy...")
    output_dict_eval_real = sess.run(
        network.get_fetch_dict('d_accuracy', 'd_loss'),
        network.get_feed_dict(inputs=s_eval, labels=np.ones([s_eval_len, 1]), input_dropout=1)
    )

    output_dict_eval_fake = sess.run(
        network.get_fetch_dict('d_accuracy', 'd_loss'),
        network.get_feed_dict(inputs=s_eval_fake, labels=np.zeros([s_eval_len, 1]), input_dropout=1)
    )
    total_accuracy = (output_dict_eval_real['d_accuracy'] + output_dict_eval_fake['d_accuracy']) / 2.0
    total_loss = (output_dict_eval_real['d_loss'] + output_dict_eval_fake['d_loss']) / 2.0
    return total_accuracy, total_loss

if __name__ == '__main__':
    main()

