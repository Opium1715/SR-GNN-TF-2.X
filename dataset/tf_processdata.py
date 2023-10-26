import csv
from functools import partial

import numpy as np
import tensorflow as tf


def data_generator(data):
    for example in data:
        yield example


def process_data(row):
    features = row[:-1]
    labels = row[-1]
    items, alias_inputs = tf.unique(features)

    vector_length = tf.shape(features)[0]
    n_nodes = tf.shape(items)[0]
    indices = tf.gather(alias_inputs, tf.stack([tf.range(vector_length - 1), tf.range(vector_length - 1) + 1],
                                               axis=0))  # Stack and stagger values
    unique_indices, _ = tf.unique(indices[0] * (vector_length + 1) + indices[1])  # unique(a*x + b)
    unique_indices = tf.sort(unique_indices)  # Sort ascending
    unique_indices = tf.stack(
        [tf.math.floordiv(unique_indices, (vector_length + 1)), tf.math.floormod(unique_indices, (vector_length + 1))],
        axis=1)  # Ungroup and stack
    unique_indices = tf.cast(unique_indices, tf.int64)

    values = tf.ones(tf.shape(unique_indices, out_type=tf.int64)[0], dtype=tf.int64)
    dense_shape = tf.cast([n_nodes, n_nodes], tf.int64)

    adj = tf.SparseTensor(indices=unique_indices, values=values, dense_shape=dense_shape)
    adj = tf.sparse.to_dense(adj)

    u_sum_in_tf = tf.math.reduce_sum(adj, 0)
    u_sum_in_tf = tf.clip_by_value(u_sum_in_tf, 1, tf.reduce_max(u_sum_in_tf))
    A_in = tf.math.divide(adj, u_sum_in_tf)

    u_sum_out_tf = tf.math.reduce_sum(adj, 1)
    u_sum_out_tf = tf.clip_by_value(u_sum_out_tf, 1, tf.reduce_max(u_sum_out_tf))
    A_out = tf.math.divide(tf.transpose(adj), u_sum_out_tf)

    mask = tf.fill(tf.shape(features), 1)
    inputs = (A_in, A_out, alias_inputs, items, mask)
    targets = labels-1

    # return A_in, A_out, alias_inputs, items, mask, labels
    return inputs, targets


def train_input_fn(batch_size, max_seq, max_n_node):
    with open("dataset/diginetica/train.csv", "r") as data_file:
        data = [list(map(int, rec)) for rec in csv.reader(data_file, delimiter=',')]

    dataset = tf.data.Dataset.from_generator(partial(data_generator, data), output_types=tf.int32)
    dataset = dataset.map(process_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # TODO: Don't forget to enable shuffle
    dataset = dataset.shuffle(100000)

    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=(
                                       ([max_n_node, max_n_node],
                                        [max_n_node, max_n_node],
                                        [max_seq],
                                        [max_n_node],
                                        [max_seq]),
                                       []),
                                   drop_remainder=True,
                                   name='batch_process'
                                   )

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def eval_input_fn(batch_size, max_seq, max_n_node):
    with open("dataset/diginetica/test.csv", "r") as data_file:
        data = [list(map(int, rec)) for rec in csv.reader(data_file, delimiter=',')]

    dataset = tf.data.Dataset.from_generator(partial(data_generator, data), output_types=tf.int32)
    dataset = dataset.map(process_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=(
                                       ([max_n_node, max_n_node],
                                        [max_n_node, max_n_node],
                                        [max_seq],
                                        [max_n_node],
                                        [max_seq]),
                                       []),
                                   drop_remainder=True,
                                   name='batch_process')

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == '__main__':
    n_node = 0
    max_seq = 0
    max_n_node = 0

    with open("diginetica/train.csv", "r") as data_file:
        data = [list(map(int, rec)) for rec in csv.reader(data_file, delimiter=',')]
        n_node = max(n_node, np.amax([np.amax(z) for z in data]) + 1)
        max_seq = max(max_seq, len(max(data, key=len)))
        max_n_node = max(max_n_node, len(max([np.unique(i) for i in data], key=len)))
        train_dataset_size = len(data)

    train_data = train_input_fn(100, max_seq, max_n_node)
    # for A_in, A_out, alias_inputs, items, mask, labels in train_data:
    #     print(A_in, A_out, alias_inputs, items, mask, labels)
    for data in train_data:
        print(data)
