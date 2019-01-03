import tensorflow as tf
from snidsa_cell import *

class SNIDSA(object):
    def __init__(self, config, A, is_training=True):

        self.num_nodes = config.num_nodes
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        # self.model = config.model

        self.learning_rate = config.learning_rate
        self.dropout = config.dropout

        self._A = tf.constant(A, dtype=tf.float32, name="adjacency_matrix")

        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable(
                "embedding", [self.num_nodes,
                    self.embedding_dim], dtype=tf.float32)

        self.placeholders()
        self.loss_mask()
        self.graph_information()
        self.recurrent_layer()
        self.cost()
        self.optimize()

    def placeholders(self):
        self.batch_size = tf.placeholder(tf.int32, None)
        self._inputs = tf.placeholder(tf.int32, [None, None]) # [batch_size, num_steps]
        self._targets = tf.placeholder(tf.int32, [None, None])
        self._seqlen = tf.placeholder(tf.int32, [None])
        self.num_steps = tf.placeholder(tf.int32, None)

    def loss_mask(self):
        self._target_mask = tf.sequence_mask(self._seqlen, dtype=tf.float32)

    def graph_information(self):
        _neighbors = tf.nn.embedding_lookup(self._A, self._inputs)
        return _neighbors

    def input_embedding(self):
        _inputs = tf.nn.embedding_lookup(self.embedding, self._inputs)
        return _inputs

    def recurrent_layer(self):
        def creat_cell():
            cell = SnidsaCell(self.hidden_dim, self.embedding)
            if self.dropout < 1:
                return tf.contrib.rnn.DropoutWrapper(cell,
                    output_keep_prob=self.dropout)
            else:
                return cell

        cells = [creat_cell() for _ in range(self.num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        emb_inputs = self.input_embedding()
        _neighbors = self.graph_information()
        _outputs, _ = tf.nn.dynamic_rnn(cell=cell,
            inputs=(emb_inputs,_neighbors), sequence_length=self._seqlen, dtype=tf.float32)

        output = tf.reshape(tf.concat(_outputs, 1), [-1, self.hidden_dim])
        softmax_w = tf.get_variable(
            "softmax_w", [self.hidden_dim, self.num_nodes], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.num_nodes], dtype=tf.float32)
        self.flat_logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
         # Reshape logits to be a 3-D tensor for sequence loss
        self._logits = tf.reshape(self.flat_logits, [self.batch_size, self.num_steps, self.num_nodes])

    def cost(self):
        crossent = tf.contrib.seq2seq.sequence_loss(
            self._logits,
            self._targets,
            self._target_mask,
            average_across_timesteps=False,
            average_across_batch=False)
        loss = tf.reduce_sum(crossent, axis=[0])
        batch_avg = tf.reduce_sum(self._target_mask, axis=[0])
        batch_avg += 1e-12  # to avoid division by 0 for all-0 weights
        loss /= batch_avg
        # Update the cost
        self.cost = tf.reduce_sum(loss)
        # Calculate negative log-likelihood
        self.nll = tf.reduce_sum(crossent, axis = [1])

        pred = tf.nn.softmax(self.flat_logits)
        self.pred = tf.reshape(pred, [self.batch_size, self.num_steps, self.num_nodes])

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optim = optimizer.minimize(self.cost)