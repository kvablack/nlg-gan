import tensorflow as tf
from model import Model
from . import layers


class LSTMNetwork(Model):
    """
    Model using an LSTMCell softmax output layers.

    Initialization hyperparameters:
        :param learning_rate (int): learning rate for optimizer
        :param dim_state (int): dimensionality of LSTM's internal state vector
        :param dim_in (int): dimensionality of LSTM input vectors
        :param sequence_length (int): the length of each sequence of inputs/outputs
        :param num_classes (int): number of output classes

    Placeholders:
        :ivar inputs: Shape [sequence_length, batch_size, dim_in] to be fed to the network
        :ivar labels: Shape [sequence_length, batch_size, num_classes] used to compute losses
        :ivar initial_state: Shape [batch_size, dim_state] to be used as initial state of LSTM cell
        :ivar initial_output: Shape [batch_size, dim_state] to be used as initial 'previous output' of LSTM cell

    Fetchable output tensors:
        :ivar outputs: List of length sequence_length with shape [batch_size, dim_state] holding the
            pre-softmax-layer raw outputs of the LSTM
        :ivar states: List of length sequence_length with shape [batch_size, dim_state] holding the state
            vectors of the LSTM at each time step
        :ivar predictions: List of length sequence_length with shape [batch_size, num_classes] holding the softmax-
            transformed probability distributions for the output class
        :ivar total_loss: total loss across entire batch and sequence
        :ivar train_step: Adagrad optimizer to minimize the total loss

    .. note::
        If truncated backpropagation is desired, then it is the responsibility of the external training loop to
        split each sequence into smaller segments to train separately so that gradients are only propagated to the
        beginning of each segment. This can be achieved by storing `outputs[-1]` and `states[-1]` outside of the
        computational graph after each segment and passing them to initial_output and initial_state at the beginning
        of the next segment.
    """

    def _gen_placeholders(self):
        self.placeholders['inputs'] = tf.placeholder(tf.float32, shape=[self.hyperparameters['sequence_length'], None,
                                                                        self.hyperparameters['dim_in']])
        self.placeholders['labels'] = tf.placeholder(tf.float32, shape=[self.hyperparameters['sequence_length'], None,
                                                                        self.hyperparameters['num_classes']])
        self.placeholders['initial_state'] = tf.placeholder(tf.float32,
                                                            shape=[None, self.hyperparameters['dim_state']])
        self.placeholders['initial_output'] = tf.placeholder(tf.float32,
                                                             shape=[None, self.hyperparameters['dim_state']])

    def _build_model(self):
        # LSTMCell(dim_in, dim_state, initial_state, initial_output)
        lstm_cell = layers.LSTMCell(self.hyperparameters['dim_in'], self.hyperparameters['dim_state'],
                                    self.placeholders['initial_state'], self.placeholders['initial_output'])

        # Final softmax regression layer from LSTM outputs to output space (maps from dim_state -> num_classes)
        final_softmax_layer = layers.FCLayer(
            [self.hyperparameters['dim_state']], self.hyperparameters['num_classes'], lambda x: x)
        # Use the identity as activation because pre-softmax logits are required for
        # tf.nn.softmax_cross_entropy_with_logits

        # Build computational graph
        self.outputs['outputs'], self.outputs['states'] = lstm_cell(self.placeholders['inputs'])
        logits = [final_softmax_layer(x) for x in self.outputs['outputs']]
        self.outputs['predictions'] = [tf.nn.softmax(x) for x in logits]

        losses = [tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)
                  for x, y in zip(logits, tf.unstack(self.placeholders['labels'], axis=0))]
        self.outputs['total_loss'] = tf.reduce_mean(losses)

        self.outputs['train_step'] = tf.train.AdagradOptimizer(self.hyperparameters['learning_rate']) \
            .minimize(self.outputs['total_loss'])
