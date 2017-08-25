import random
import tensorflow as tf
import numpy as np
from plot import Plotter
from basic_lstm.model import LSTMNetwork

LEARNING_RATE = 0.3
DIM_STATE = 30
DIM_IN = 1
SEQUENCE_LENGTH = 50
BATCH_SIZE = 30
EPOCH_LENGTH = 10000


def main():
    plotter = Plotter()
    plot_counter = 0
    with tf.Session() as sess:
        network = LSTMNetwork(learning_rate=LEARNING_RATE, dim_state=DIM_STATE, dim_in=DIM_IN, sequence_length=SEQUENCE_LENGTH, num_classes=10)
        sess.run(tf.global_variables_initializer())

        # fig, ax = plt.subplots()
        # fig.canvas.draw()
        # line, = plt.plot(loss_list)

        for epoch in range(1000):
            print("Epoch %d" % epoch)
            # generate random sequence of integers
            data = np.array([random.randint(0, 9) for i in range(EPOCH_LENGTH * BATCH_SIZE)])
            # split into batches
            data = data.reshape([EPOCH_LENGTH, BATCH_SIZE, 1])
            # encode as one-hot vectors
            labels = np.zeros([EPOCH_LENGTH, BATCH_SIZE, 10])
            for i in range(EPOCH_LENGTH):
                labels[i] = np.eye(10)[data[i].reshape(-1)]

            # split into sequences of SEQUENCE_LENGTH
            data = np.array_split(data, EPOCH_LENGTH / SEQUENCE_LENGTH, axis=0)
            labels = np.array_split(labels, EPOCH_LENGTH / SEQUENCE_LENGTH, axis=0)

            # initialize state to zeros
            state, output = np.zeros([BATCH_SIZE, DIM_STATE]), np.zeros([BATCH_SIZE, DIM_STATE])
            for sequence in range(len(data)):
                output_dict = sess.run(
                    network.get_fetch_dict('outputs', 'states', 'total_loss', 'train_step'),
                    network.get_feed_dict(inputs=data[sequence], labels=labels[sequence], initial_state=state, initial_output=output)
                )
                state, output = output_dict['states'][-1], output_dict['outputs'][-1]

                if sequence % 10 == 0:
                    print("Finished training sequence %d" % sequence)
                    # ln.set_data(np.arange(len(loss_list)), np.array(loss_list))
                    # ax.relim()
                    # ax.autoscale_view()
                    # fig.canvas.draw()
                    # fig.canvas.flush_events()
                    # plt.pause(0.000000000001)
                    plotter.plot(plot_counter, output_dict['total_loss'])
                    plot_counter += 10

if __name__ == '__main__':
    main()

