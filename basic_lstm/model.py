import tensorflow as tf
from model import Model
from . import layers


class NlpGan(Model):
    """
    NLP GAN using two LSTM networks for the generator and discriminator.

    Initialization hyperparameters:
        :param learning_rate (int): learning rate for optimizer
        :param dim_state (int): dimensionality of LSTM's internal state vector
        :param dim_in (int): dimensionality of LSTM input vectors
        :param sequence_length (int): the length of each sequence of inputs

    Placeholders:
        :ivar inputs: Shape [sequence_length, batch_size, dim_in] to be fed to the network
        :ivar initial_state: Shape [batch_size, dim_state] to be used as initial state of LSTM cell
        :ivar initial_output: Shape [batch_size, dim_state] to be used as initial 'previous output' of LSTM cell

    Fetchable output tensors:
        :ivar d_loss: discriminator loss across entire batch and sequence
        :ivar d_train: Adam optimizer for discriminator
    """

    def _gen_placeholders(self):
        self.placeholders['inputs'] = tf.placeholder(tf.float32, shape=[self.hyperparameters['sequence_length'], None,
                                                                        self.hyperparameters['dim_in']])

        self.placeholders['initial_state'] = tf.placeholder(tf.float32,
                                                            shape=[None, self.hyperparameters['dim_state']])
        self.placeholders['initial_output'] = tf.placeholder(tf.float32,
                                                             shape=[None, self.hyperparameters['dim_state']])

        self.placeholders['labels'] = tf.placeholder(tf.float32, shape=[None, 1])

    def _build_model(self):
        # Discriminator LSTM cell
        discriminator = layers.LSTMCell(self.hyperparameters['dim_in'], self.hyperparameters['dim_state'],
                                    self.placeholders['initial_state'], self.placeholders['initial_output'])

        # Discriminator final dense layer for binary classification
        final_dense_layer = layers.FCLayer(
            [self.hyperparameters['dim_state']], 1, lambda x: x) # Use the identity as activation because logits are
                                                                 # required for tf.nn.sigmoid_cross_entropy_with_logits

        # Build computational graph
        d_outputs, _ = discriminator(self.placeholders['inputs'])
        d_logits = final_dense_layer(d_outputs[-1])

        d_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=self.placeholders['labels'])
        self.outputs['d_loss'] = tf.reduce_mean(d_losses)

        self.outputs['d_train'] = tf.train.AdamOptimizer(self.hyperparameters['learning_rate']) \
            .minimize(self.outputs['d_loss'])

        d_predictions = tf.round(tf.nn.sigmoid(d_logits))
        d_correct = tf.equal(d_predictions, self.placeholders['labels'])
        self.outputs['d_accuracy'] = tf.reduce_mean(tf.cast(d_correct, tf.float32))
