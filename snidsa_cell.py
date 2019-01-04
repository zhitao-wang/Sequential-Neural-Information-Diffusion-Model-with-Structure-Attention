#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

class SnidsaCell(RNNCell):
    """ Recurrent Unit Cell for SNIDSA."""

    def __init__(self, num_units, feat_in_matrix, activation=None, reuse=None):
        self._num_units = num_units
        self._num_nodes = int(feat_in_matrix.shape[0])
        self._feat_in_matrix = feat_in_matrix
        self._feat_in = int(feat_in_matrix.shape[1])
        self._activation = activation or tf.tanh

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        X = inputs[0]
        A = inputs[1]
        feat_in = self._feat_in
        feat_out = self._num_units
        num_nodes = self._num_nodes
        feat_in_matrix = self._feat_in_matrix

        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("Attentions"):
                struc, x = struc_atten(X, feat_in_matrix, A, feat_in)
            with tf.variable_scope("c_inputs"):
                xs_c = linear([x, struc], feat_out, False)
            with tf.variable_scope("h_inputs"):
                xs_h = linear([x, struc], feat_out, False)
            with tf.variable_scope("Gate"):
                concat = tf.sigmoid(
                    linear([X, struc], 2 * feat_out, True))
                if tf.__version__ == "0.12.1":
                    f, r = tf.split(1, 2, concat)
                else:
                    f, r = tf.split(axis=1, num_or_size_splits=2, value=concat)

            c = f * state + (1 - f) * xs_c

            # highway connection
            h = r * self._activation(c) + (1 - r) * xs_h

        return h, c


def struc_atten(X, feat_in_matrix, A, feat_out):
    batch_size = tf.shape(X)[0]
    num_nodes = tf.shape(feat_in_matrix)[0]
    with tf.variable_scope("linear_transf"):
        linear_transf_X = linear([X], feat_out, False)
        tf.get_variable_scope().reuse_variables()
        linear_transf_G = linear([feat_in_matrix], feat_out, False)
    with tf.variable_scope("strcuture_attention"):
        Wa = tf.get_variable("Wa", [2*feat_out, 1], dtype=tf.float32,
                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

        # Repeat feature vectors of input: [[1], [2]] becomes [[1], [1], [2], [2]]
        repeated = tf.reshape(tf.tile(linear_transf_X, (1, num_nodes)), (batch_size * num_nodes, feat_out))  # (BN x F')
        # Tile feature vectors of full graph: [[1], [2]] becomes [[1], [2], [1], [2]]
        tiled = tf.tile(linear_transf_G, (batch_size, 1))  # (BN x F')
        # Build combinations
        combinations = tf.concat([repeated, tiled],1)  # (BN x 2F')
        combination_slices = tf.reshape(combinations, (batch_size, -1, 2 * feat_out))  # (B x N x 2F')

        dense = tf.squeeze(tf.contrib.keras.backend.dot(combination_slices, Wa), -1)  
        # 降维成 B X N
        comparison = tf.equal(A, tf.constant(0, dtype=tf.float32))
        mask = tf.where(comparison, tf.ones_like(A) * -10e9, tf.zeros_like(A))
        masked = dense + mask

        struc_att = tf.nn.softmax(masked)  # (B x N)
        struc_att = tf.nn.dropout(struc_att, 1)  # Apply dropout to normalized attention coefficients (B x N)

        # Linear combination with neighbors' features
        struc = tf.matmul(struc_att, linear_transf_G)  # (B x F')

        struc = tf.nn.elu(struc)

    return struc, linear_transf_X



def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                "Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError(
                "Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
    return res + bias_term
