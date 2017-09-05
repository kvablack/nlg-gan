import tensorflow as tf
import numpy as np
from plot import Plotter
from nltk.corpus import gutenberg
from basic_lstm.model import NlpGan
import utils

SAVE_NAME = 'GAN_10_NORMING'

LEARNING_RATE = 0.000007
D_DIM_STATE = 150
G_DIM_STATE = 600
WORD_DIM = 300
SEQUENCE_LENGTH = 10
BATCH_SIZE = 100
D_KEEP_PROB = 0.5
MAX_LOSS_DIFF = 0.05
INSTANCE_VARIANCE = 0.15


def nearest_neighbor(words, wordvecs_norm, wordvec_norm):
    similarities = np.dot(wordvecs_norm, wordvec_norm.reshape(-1)).reshape(-1)
    index = np.argmax(similarities)
    return reverse_lookup(words, index), similarities[index]


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

    wordvecs_norm = wordvecs / np.linalg.norm(wordvecs, axis=1).reshape(-1, 1)

    print("Loading corpus...")
    # Convert corpus into normed wordvecs, replacing any words not in vocab with zero vector
    sentences = [[wordvecs_norm[words[clean_word(word)]] if clean_word(word) in words.keys() else np.zeros(WORD_DIM)
                  for word in sentence]
                 for sentence in gutenberg.sents()]

    print("Processing corpus...")
    # Pad sentences shorter than SEQUENCE_LENGTH with zero vectors and truncate sentences longer than SEQUENCE_LENGTH
    s_train = list(map(pad_or_truncate, sentences))

    np.random.shuffle(s_train)

    # Truncate to multiple of BATCH_SIZE
    s_train = s_train[:int(len(s_train) / BATCH_SIZE) * BATCH_SIZE]

    s_train_idxs = np.arange(len(s_train))

    print("Generating graph...")
    network = NlpGan(learning_rate=LEARNING_RATE, d_dim_state=D_DIM_STATE, g_dim_state=G_DIM_STATE,
                     dim_in=WORD_DIM, sequence_length=SEQUENCE_LENGTH)

    plotter = Plotter([2, 1], "Loss", "Accuracy")
    plotter.plot(0, 0, 0, 0)
    plotter.plot(0, 0, 0, 1)
    plotter.plot(0, 0, 1, 0)
    plotter.plot(0, 1, 1, 0)

    #d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #eval(sess, network, words, wordvecs_norm, saver)

        sess.run(tf.global_variables_initializer())
        #resume(sess, saver, plotter, "GAN_9_SEQUENCELENGTH_10", 59)

        d_loss, g_loss = 0.0, 0.0
        for epoch in range(0, 10000000):
            print("Epoch %d" % epoch)

            np.random.shuffle(s_train_idxs)
            for batch in range(int(len(s_train_idxs) / BATCH_SIZE)):
                # select next random batch of sentences
                s_batch_real = [s_train[x] for x in s_train_idxs[batch:batch + BATCH_SIZE]] # shape (BATCH_SIZE, SEQUENCE_LENGTH, WORD_DIM)

                # reshape to (SEQUENCE_LENGTH, BATCH_SIZE, WORD_DIM) while preserving sentence order
                s_batch_real = np.array(s_batch_real).swapaxes(0, 1)

                if d_loss - g_loss > MAX_LOSS_DIFF and False:
                    output_dict = sess.run(
                        network.get_fetch_dict('d_loss', 'd_train', 'g_loss'),
                        network.get_feed_dict(inputs=s_batch_real, input_dropout=D_KEEP_PROB)
                    )
                elif g_loss - d_loss > MAX_LOSS_DIFF and False:
                    output_dict = sess.run(
                        network.get_fetch_dict('d_loss', 'g_loss', 'g_train'),
                        network.get_feed_dict(inputs=s_batch_real, input_dropout=D_KEEP_PROB)
                    )
                else:
                    output_dict = sess.run(
                        network.get_fetch_dict('d_loss', 'd_train', 'g_loss', 'g_train'),
                        network.get_feed_dict(inputs=s_batch_real, input_dropout=D_KEEP_PROB,
                                              instance_variance=INSTANCE_VARIANCE)
                    )

                d_loss, g_loss = output_dict['d_loss'], output_dict['g_loss']

                if batch % 10 == 0:
                    print("Finished training batch %d / %d" % (batch, int(len(s_train) / BATCH_SIZE)))
                    print("Discriminator Loss: %f" % output_dict['d_loss'])
                    print("Generator Loss: %f" % output_dict['g_loss'])
                    plotter.plot(epoch + (batch / int(len(s_train) / BATCH_SIZE)), d_loss, 0, 0)
                    plotter.plot(epoch + (batch / int(len(s_train) / BATCH_SIZE)), g_loss, 0, 1)

                if batch % 100 == 0:
                    eval = sess.run(
                        network.get_fetch_dict('g_outputs', 'd_accuracy'),
                        network.get_feed_dict(inputs=s_batch_real, input_dropout=1.0,
                                              instance_variance=INSTANCE_VARIANCE)
                    )
                    # reshape g_outputs to (BATCH_SIZE, SEQUENCE_LENGTH, WORD_DIM) while preserving sentence order
                    generated = eval['g_outputs'].swapaxes(0, 1)
                    for sentence in generated[:3]:
                        for wordvec in sentence:
                            norm = np.linalg.norm(wordvec)
                            word, similarity = nearest_neighbor(words, wordvecs_norm, wordvec / norm)
                            print("{}({:4.2f})".format(word, similarity), end=' ')
                        print('\n---------')
                    print("Total Accuracy: %f" % eval['d_accuracy'])
                    plotter.plot(epoch + (batch / int(len(s_train) / BATCH_SIZE)), eval['d_accuracy'], 1, 0)

            saver.save(sess, './checkpoints/{}.ckpt'.format(SAVE_NAME),
                       global_step=epoch)
            plotter.save(SAVE_NAME)


def resume(sess, saver, plotter, save_name, epoch):
    saver.restore(sess, "./checkpoints\\{}.ckpt-{}".format(save_name, epoch))
    plotter.load(save_name)


def eval(sess, network, words, wordvecs_norm, saver):
    saver.restore(sess, "./checkpoints\\GAN_1.ckpt-28")
    while True:
        output_dict = sess.run(
            network.get_fetch_dict('g_outputs'),
            network.get_feed_dict(inputs=np.zeros([SEQUENCE_LENGTH, 10, WORD_DIM]))
        )

        generated = output_dict['g_outputs'].swapaxes(0, 1)
        for sentence in generated:
            print("----------------------------------------------------------")
            for wordvec in sentence:
                print("{}    --    {}".format(np.linalg.norm(wordvec),
                                              nearest_neighbor(words, wordvecs_norm, wordvec / np.linalg.norm(wordvec))))


if __name__ == '__main__':
    main()

