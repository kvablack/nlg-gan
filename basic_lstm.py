import tensorflow as tf
import numpy as np
from plot import Plotter
from nltk.corpus import gutenberg
from basic_lstm.model import NlpGan
import utils

LEARNING_RATE = 0.0002
DIM_STATE = 100
WORD_DIM = 300
SEQUENCE_LENGTH = 50
BATCH_SIZE = 60


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


def gen_fake_sentences(sentences_rev, n):
    fake_batch = [sentences_rev[x] for x in np.random.randint(0, len(sentences_rev), n)]
    fake_batch = np.array(fake_batch).reshape(-1, WORD_DIM).reshape(SEQUENCE_LENGTH, n, WORD_DIM, order='F')
    return fake_batch


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
    sentences_rev = [sentence[::-1] for sentence in sentences]
    # Pad sentences shorter than SEQUENCE_LENGTH with zero vectors and truncate sentences longer than SEQUENCE_LENGTH
    sentences = list(map(pad_or_truncate, sentences))
    sentences_rev = list(map(pad_or_truncate, sentences_rev))

    # Split up into train/eval sections, 97% train and 3% eval
    s_train, s_eval = sentences[:int(len(sentences) * 0.97)], sentences[int(len(sentences) * 0.97):]
    # Truncate to multiple of BATCH_SIZE
    s_train = s_train[:int(len(s_train) / BATCH_SIZE) * BATCH_SIZE]

    s_eval_len = len(s_eval)
    # reshape to (SEQUENCE_LENGTH, BATCH_SIZE, WORD_DIM) while preserving sentence order; BATCH_SIZE is entire dataset
    # in this case
    s_eval = np.array(s_eval).reshape(-1, WORD_DIM).reshape(SEQUENCE_LENGTH, -1, WORD_DIM, order='F')

    s_train_idxs = np.arange(len(s_train))

    print("Generating graph...")
    network = NlpGan(learning_rate=LEARNING_RATE, dim_state=DIM_STATE, dim_in=WORD_DIM, sequence_length=SEQUENCE_LENGTH)

    plotter = Plotter(2, "Loss", "Accuracy")
    plotter.plot(0, 0, 0)
    plotter.plot(0, 0, 1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(100000):
            print("Epoch %d" % epoch)

            np.random.shuffle(s_train_idxs)
            for batch in range(int(len(s_train_idxs) / BATCH_SIZE)):
                # select next random batch of sentences
                s_batch_real = [s_train[x] for x in s_train_idxs[batch:batch + BATCH_SIZE]] # shape (BATCH_SIZE, SEQUENCE_LENGTH, WORD_DIM)

                # reshape to (SEQUENCE_LENGTH, BATCH_SIZE, WORD_DIM) while preserving sentence order
                s_batch_real = np.array(s_batch_real).reshape(-1, WORD_DIM).reshape(SEQUENCE_LENGTH, BATCH_SIZE, WORD_DIM, order='F')

                # real text
                output_dict_real = sess.run(
                    network.get_fetch_dict('d_loss', 'd_train'),
                    network.get_feed_dict(inputs=s_batch_real, labels=np.ones([BATCH_SIZE, 1]),
                                          initial_state=np.zeros([BATCH_SIZE, DIM_STATE]),
                                          initial_output=np.zeros([BATCH_SIZE, DIM_STATE]))
                )

                s_batch_fake = gen_fake_sentences(sentences_rev, BATCH_SIZE)
                # fake text
                output_dict_fake = sess.run(
                    network.get_fetch_dict('d_loss', 'd_train'),
                    network.get_feed_dict(inputs=s_batch_fake, labels=np.zeros([BATCH_SIZE, 1]),
                                          initial_state=np.zeros([BATCH_SIZE, DIM_STATE]),
                                          initial_output=np.zeros([BATCH_SIZE, DIM_STATE]))
                )

                total_loss = (output_dict_real['d_loss'] + output_dict_fake['d_loss']) / 2.0
                if batch % 10 == 0:
                    print("Finished training batch %d / %d" % (batch, int(len(s_train_idxs) / BATCH_SIZE)))
                    print("Total Loss: %f" % total_loss)
                    plotter.plot(epoch + (batch / int(len(s_train_idxs) / BATCH_SIZE)), total_loss, 0)

            # after epoch is finished, evaluate accuracy
            print("Evaluating accuracy...")
            output_dict_eval_real = sess.run(
                network.get_fetch_dict('d_accuracy'),
                network.get_feed_dict(inputs=s_eval, labels=np.ones([s_eval_len, 1]),
                                      initial_state=np.zeros([s_eval_len, DIM_STATE]),
                                      initial_output=np.zeros([s_eval_len, DIM_STATE]))
            )
            s_eval_fake = gen_fake_sentences(sentences_rev, s_eval_len)
            output_dict_eval_fake = sess.run(
                network.get_fetch_dict('d_accuracy'),
                network.get_feed_dict(inputs=s_eval_fake, labels=np.zeros([s_eval_len, 1]),
                                      initial_state=np.zeros([s_eval_len, DIM_STATE]),
                                      initial_output=np.zeros([s_eval_len, DIM_STATE]))
            )
            total_accuracy = (output_dict_eval_real['d_accuracy'] + output_dict_eval_fake['d_accuracy']) / 2.0
            print("Total Accuracy: %f" % total_accuracy)
            plotter.plot(epoch + 1, total_accuracy, 1)

            if epoch % 10 == 0:
                saver.save(sess, './checkpoints/test.ckpt',
                           global_step=epoch)

if __name__ == '__main__':
    main()

