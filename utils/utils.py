# import pickle
#
# import numpy as np
# import tensorflow as tf
#
#
# # def build_graph(train_data):
# #     graph = nx.DiGraph()
# #     for seq in train_data:
# #         for i in range(len(seq) - 1):
# #             if graph.get_edge_data(seq[i], seq[i + 1]) is None:
# #                 weight = 1
# #             else:
# #                 weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
# #             graph.add_edge(seq[i], seq[i + 1], weight=weight)
# #     for node in graph.nodes:
# #         sum = 0
# #         for j, i in graph.in_edges(node):
# #             sum += graph.get_edge_data(j, i)['weight']
# #         if sum != 0:
# #             for j, i in graph.in_edges(i):
# #                 graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
# #     return graph
#
#
# def data_masks(all_usr_pois, item_tail):
#     us_lens = [len(upois) for upois in all_usr_pois]
#     len_max = max(us_lens)
#     us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
#     us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
#     return us_pois, us_msks, len_max
#
#
# def split_validation(train_set, valid_portion):
#     train_set_x, train_set_y = train_set
#     n_samples = len(train_set_x)
#     sidx = np.arange(n_samples, dtype='int32')
#     np.random.shuffle(sidx)
#     n_train = int(np.round(n_samples * (1. - valid_portion)))
#     valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
#     valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
#     train_set_x = [train_set_x[s] for s in sidx[:n_train]]
#     train_set_y = [train_set_y[s] for s in sidx[:n_train]]
#
#     return (train_set_x, train_set_y), (valid_set_x, valid_set_y)
#
#
# class TFData:
#     def __init__(self, file_path, batch_size, shuffle=False):
#         self.batch_size = batch_size
#         self.data_file = pickle.load(open(file_path, "rb"))
#
#         self.dataset_size = len(self.data_file[0])
#         self.max_seq = len(max(self.data_file[0], key=len))
#         self.max_n_node = len(max([np.unique(i) for i in self.data_file[0]], key=len))
#
#         dataset = tf.data.Dataset.from_generator(self.data_generator, output_types=(tf.int32, tf.int32))
#         dataset = dataset.map(self.get_adj)
#         dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=(
#             [self.max_n_node, self.max_n_node],
#             [self.max_n_node, self.max_n_node],
#             [self.max_seq],
#             [self.max_seq],
#             [self.max_seq],
#             []))
#
#         if shuffle:
#             dataset = dataset.shuffle(100000)
#
#         self.dataset = dataset.prefetch(batch_size)
#
#     def data_generator(self):
#         features, labels = self.data_file
#         for i in range(len(features)):
#             yield features[i], labels[i]
#
#     def get_adj(self, features, labels):
#         items, alias_inputs = tf.unique(features)
#
#         vector_length = tf.shape(features)[0]
#         n_nodes = tf.shape(items)[0]
#         indices = tf.gather(alias_inputs, tf.stack([tf.range(vector_length - 1), tf.range(vector_length - 1) + 1],
#                                                    axis=0))  # Stack and stagger values
#         unique_indices, _ = tf.unique(indices[0] * (vector_length + 1) + indices[1])  # unique(a*x + b)
#         unique_indices = tf.sort(unique_indices)  # Sort ascending
#         unique_indices = tf.stack(
#             [tf.floor_div(unique_indices, (vector_length + 1)), tf.floormod(unique_indices, (vector_length + 1))],
#             axis=1)  # Ungroup and stack
#         unique_indices = tf.cast(unique_indices, tf.int64)
#
#         values = tf.ones(tf.shape(unique_indices, out_type=tf.int64)[0], dtype=tf.int64)
#         dense_shape = tf.cast([n_nodes, n_nodes], tf.int64)
#
#         adj = tf.SparseTensor(indices=unique_indices, values=values, dense_shape=dense_shape)
#         adj = tf.sparse.to_dense(adj)
#
#         u_sum_in_tf = tf.math.reduce_sum(adj, 0)
#         u_sum_in_tf = tf.clip_by_value(u_sum_in_tf, 1, tf.reduce_max(u_sum_in_tf))
#         A_in = tf.math.divide(adj, u_sum_in_tf)
#
#         u_sum_out_tf = tf.math.reduce_sum(adj, 1)
#         u_sum_out_tf = tf.clip_by_value(u_sum_out_tf, 1, tf.reduce_max(u_sum_out_tf))
#         A_out = tf.math.divide(tf.transpose(adj), u_sum_out_tf)
#
#         mask = tf.fill(tf.shape(features), 1)
#
#         return A_in, A_out, alias_inputs, items, mask, labels
#
#     def get_batch(self):
#         for i in self.dataset:
#             yield i
#
#
# class Data:
#     def __init__(self, data, sub_graph=False, method='ggnn', sparse=False, shuffle=False):
#         inputs = data[0]
#         inputs, mask, len_max = data_masks(inputs, [0])
#         self.inputs = np.asarray(inputs)
#         self.mask = np.asarray(mask)
#         self.len_max = len_max
#         self.targets = np.asarray(data[1])
#         self.length = len(inputs)
#         self.shuffle = shuffle
#         self.sub_graph = sub_graph
#         self.sparse = sparse
#         self.method = method
#
#     def generate_batch(self, batch_size):
#         if self.shuffle:
#             shuffled_arg = np.arange(self.length)
#             np.random.shuffle(shuffled_arg)
#             self.inputs = self.inputs[shuffled_arg]
#             self.mask = self.mask[shuffled_arg]
#             self.targets = self.targets[shuffled_arg]
#         n_batch = int(self.length / batch_size)
#         if self.length % batch_size != 0:
#             n_batch += 1
#         slices = np.split(np.arange(n_batch * batch_size), n_batch)
#         slices[-1] = np.arange(self.length - batch_size, self.length)
#         return slices
#
#     def get_slice(self, index):
#         if 1:
#             items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
#             for u_input in self.inputs[index]:
#                 n_node.append(len(np.unique(u_input)))
#             max_n_node = np.max(n_node)
#             if self.method == 'ggnn':
#                 for u_input in self.inputs[index]:
#                     node = np.unique(u_input)
#                     items.append(node.tolist() + (max_n_node - len(node)) * [0])
#                     u_A = np.zeros((max_n_node, max_n_node))
#                     for i in np.arange(len(u_input) - 1):
#                         if u_input[i + 1] == 0:
#                             break
#                         u = np.where(node == u_input[i])[0][0]
#                         v = np.where(node == u_input[i + 1])[0][0]
#                         u_A[u][v] = 1
#                     u_sum_in = np.sum(u_A, 0)
#                     u_sum_in[np.where(u_sum_in == 0)] = 1
#                     u_A_in = np.divide(u_A, u_sum_in)
#                     u_sum_out = np.sum(u_A, 1)
#                     u_sum_out[np.where(u_sum_out == 0)] = 1
#                     u_A_out = np.divide(u_A.transpose(), u_sum_out)
#
#                     A_in.append(u_A_in)
#                     A_out.append(u_A_out)
#                     alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
#                 return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]
#             elif self.method == 'gat':
#                 A_in = []
#                 A_out = []
#                 for u_input in self.inputs[index]:
#                     node = np.unique(u_input)
#                     items.append(node.tolist() + (max_n_node - len(node)) * [0])
#                     u_A = np.eye(max_n_node)
#                     for i in np.arange(len(u_input) - 1):
#                         if u_input[i + 1] == 0:
#                             break
#                         u = np.where(node == u_input[i])[0][0]
#                         v = np.where(node == u_input[i + 1])[0][0]
#                         u_A[u][v] = 1
#                     A_in.append(-1e9 * (1 - u_A))
#                     A_out.append(-1e9 * (1 - u_A.transpose()))
#                     alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
#                 return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]
#
#         # else:
#         #     return self.inputs[index], self.mask[index], self.targets[index]
