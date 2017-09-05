import tensorflow as tf
from model import Model
from . import layers


class NlpGan(Model):
    """
    NLP GAN using two LSTM networks for the generator and discriminator.

    Initialization hyperparameters:
        :param learning_rate (int): learning rate for optimizer
        :param d_dim_state (int): dimensionality of discriminator internal state vector
        :param g_dim_state (int): dimensionality of generator internal state vector
        :param dim_in (int): dimensionality of LSTM input vectors
        :param sequence_length (int): the length of each sequence of inputs

    Placeholders:
        :ivar inputs: Shape [sequence_length, batch_size, dim_in] to be fed to the network
        :ivar input_dropout: keep_prob for input dropout of discriminator only
        :ivar instance_variance: variance of Gaussian noise to be added to discriminator inputs

    Fetchable output tensors:
        :ivar d_loss: discriminator loss across entire batch and sequence
        :ivar d_train: Adam optimizer for discriminator
        :ivar g_loss; generator loss
        :ivar g_train: Adam optimizer for generator
        :ivar g_outputs: shape [sequence_length, batch_size, dim_in] holding the generator's generated samples
        :ivar d_accuracy: combined accuracy of discriminator for separating real from fake
    """

    def _gen_placeholders(self):
        self.placeholders['inputs'] = tf.placeholder(tf.float32, shape=[self.hyperparameters['sequence_length'], None,
                                                                        self.hyperparameters['dim_in']])

        self.placeholders['labels'] = tf.placeholder(tf.float32, shape=[None, 1])

        self.placeholders['input_dropout'] = tf.placeholder(tf.float32)

        self.placeholders['instance_variance'] = tf.placeholder(tf.float32)

    def _build_model(self):
        batch_size = tf.shape(self.placeholders['inputs'])[1]

        with tf.name_scope("discriminator"):
            d_initial_state = tf.Variable(tf.fill([self.hyperparameters['d_dim_state']], 0.0))
            d_initial_output = tf.Variable(tf.fill([self.hyperparameters['d_dim_state']], 0.0))

            # expand the initial vectors into the batch size by expanding the dimensions and tiling by runtime batch size
            d_initial_state_batch = tf.tile(tf.expand_dims(d_initial_state, 0), [batch_size, 1])
            d_initial_output_batch = tf.tile(tf.expand_dims(d_initial_output, 0), [batch_size, 1])

            # Discriminator LSTM cell
            discriminator = layers.LSTMCell(self.hyperparameters['dim_in'], self.hyperparameters['d_dim_state'],
                                            self.placeholders['input_dropout'],
                                            d_initial_state_batch, d_initial_output_batch)

            # Discriminator final dense layer for binary classification
            d_readout = layers.FCLayer([self.hyperparameters['d_dim_state']], 1,
                                       lambda x: x)  # Use the identity as activation because logits are
                                                     # required for tf.nn.sigmoid_cross_entropy_with_logits

        with tf.name_scope("generator"):
            g_initial_state = tf.Variable(tf.fill([self.hyperparameters['g_dim_state']], 0.0))
            g_initial_output = tf.Variable(tf.fill([self.hyperparameters['g_dim_state']], 0.0))

            # expand the initial vectors into the batch size by expanding the dimensions and tiling by runtime batch size
            g_initial_state_batch = tf.tile(tf.expand_dims(g_initial_state, 0), [batch_size, 1])
            g_initial_output_batch = tf.tile(tf.expand_dims(g_initial_output, 0), [batch_size, 1])

            # Generator LSTM cell with no dropout
            generator = layers.LSTMCell(self.hyperparameters['dim_in'], self.hyperparameters['g_dim_state'],
                                        tf.constant(1.0), g_initial_state_batch, g_initial_output_batch)

            # Generator final dense layer for transforming to dimensionality of word vectors
            g_readout = layers.FCLayer([self.hyperparameters['g_dim_state']], self.hyperparameters['dim_in'],
                                       tf.nn.tanh)

        # Build computational graph

        # random noise input for generator
        z = [tf.random_normal([batch_size, self.hyperparameters['dim_in']], 0, 1)
             for _ in range(self.hyperparameters['sequence_length'])]

        # generator graph
        g_outputs, _ = generator(z)
        g_outputs = [tf.nn.l2_normalize(g_readout(x), 1) for x in g_outputs]

        # add instance noise
        d_inputs_fake = [x + tf.random_normal([batch_size, self.hyperparameters['dim_in']], 0,
                                         self.placeholders['instance_variance'])
                         for x in g_outputs]
        d_inputs_real = [x + tf.random_normal([batch_size, self.hyperparameters['dim_in']], 0,
                                              self.placeholders['instance_variance'])
                         for x in tf.unstack(self.placeholders['inputs'], axis=0)]


        # discriminator graph
        d_outputs_fake, _ = discriminator(d_inputs_fake)
        d_outputs_real, _ = discriminator(d_inputs_real)

        d_logits_fake = d_readout(d_outputs_fake[-1])
        d_logits_real = d_readout(d_outputs_real[-1])


        # generator loss
        self.outputs['g_loss'] = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

        # discriminator loss
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake))
        d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_fake))

        self.outputs['d_loss'] = (tf.reduce_mean(d_loss_fake) + tf.reduce_mean(d_loss_real)) / 2.0

        # get Variables from each graph
        d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]

        # optimizers
        self.outputs['d_train'] = tf.train.AdamOptimizer(self.hyperparameters['learning_rate']) \
            .minimize(self.outputs['d_loss'], var_list=d_vars)
        self.outputs['g_train'] = tf.train.AdamOptimizer(self.hyperparameters['learning_rate']) \
            .minimize(self.outputs['g_loss'], var_list=g_vars)

        # for evaluation purposes
        self.outputs['g_outputs'] = tf.stack(g_outputs)

        d_predictions_fake = tf.round(tf.nn.sigmoid(d_logits_fake))
        d_predictions_real = tf.round(tf.nn.sigmoid(d_logits_real))

        d_correct_fake = tf.equal(d_predictions_fake, tf.zeros_like(d_logits_fake))
        d_correct_real = tf.equal(d_predictions_real, tf.ones_like(d_logits_real))

        d_accuracy_fake = tf.reduce_mean(tf.cast(d_correct_fake, tf.float32))
        d_accuracy_real = tf.reduce_mean(tf.cast(d_correct_real, tf.float32))
        self.outputs['d_accuracy'] = (d_accuracy_fake + d_accuracy_real) / 2.0

