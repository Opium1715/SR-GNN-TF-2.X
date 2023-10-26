import csv

import numpy
import numpy as np
import tensorflow as tf
from model import MyModel

if __name__ == '__main__':
    # tf.stack([tf.range(self.batch_size), tf.cast(rm, tf.int32) - 1], axis=1)
    # a = tf.random.uniform([100], maxval=150, dtype=tf.int32)
    # result = tf.stack(values=[tf.range(100), a], axis=1)
    # print(result)
    # a = tf.random.uniform([1, 2, 3, 4], dtype=tf.float32)
    # print(tf.keras.activations.sigmoid(a))
    # print(tf.nn.sigmoid(a))
    # tf.keras.
    # inputs = (tf.random.uniform(shape=[100, 25, 25]),
    #           tf.random.uniform(shape=[100, 25, 25]),
    #           tf.random.uniform(shape=[100, 34]),
    #           tf.random.uniform(shape=[100, 25]),
    #           tf.random.uniform(shape=[100, 34]))
    # print(inputs[3].shape)
    # indeices = tf.random.uniform(shape=(5, 34), maxval=4434, dtype=tf.int32)
    # mask = tf.random.uniform(shape=(5, 34))
    # mask = tf.reduce_sum(mask, axis=1)
    # print(indeices)
    # a = tf.stack([tf.range(5), tf.cast(mask, dtype=tf.int32) - 1], axis=1)
    # print(a)
    # out = tf.gather_nd(indices=a,
    #                    params=indeices)
    # print(out)
    # index = tf.range(20)
    # label = 5
    # print(np.asarray(label == index).nonzero())
    model = MyModel(hidden_size=100, out_size=100, batch_size=100, n_node=36)
    a = model.trainable_variables
    b = model.trainable_weights
    c = model.ggnn_layer.trainable_weights
    d = model.ggnn_layer.trainable_weights
    print(b)
