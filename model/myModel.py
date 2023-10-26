import math

import keras.regularizers
import tensorflow as tf


class GGNN(tf.keras.layers.Layer):
    def __init__(self, batch_size, hidden_size, out_size, **kwargs):
        super().__init__(**kwargs)
        self.B = None
        self.b_out = None
        self.W_out = None
        self.b_in = None
        self.W_in = None
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.out_size = out_size
        self.stdv = 1.0 / math.sqrt(self.hidden_size)
        self.cell = tf.keras.layers.GRUCell(self.hidden_size, use_bias=False)

    # 构造方法 -- 初始化该处理层中的一些权重参数
    def build(self, input_shape):
        input_shape = self.hidden_size
        l2 = keras.regularizers.L2(l2=1e-5)
        init = tf.random_uniform_initializer(minval=-self.stdv,
                                             maxval=self.stdv)
        self.W_in = self.add_weight(shape=(input_shape, input_shape),
                                    initializer=init,
                                    trainable=True,
                                    name='W_in',
                                    # regularizer=l2,
                                    dtype=tf.float32)
        self.b_in = self.add_weight(shape=(input_shape,),
                                    initializer=init,
                                    trainable=True,
                                    name='b_in',
                                    # regularizer=l2,
                                    dtype=tf.float32)
        self.W_out = self.add_weight(shape=(input_shape, input_shape),
                                     initializer=init,
                                     trainable=True,
                                     # regularizer=l2,
                                     name='W_out',
                                     dtype=tf.float32)
        self.b_out = self.add_weight(shape=(input_shape,),
                                     initializer=init,
                                     trainable=True,
                                     # regularizer=l2,
                                     name='b_out',
                                     dtype=tf.float32)

    # 执行方法
    def call(self, inputs, *args, **kwargs):
        # inputs 直接就是 [fin_state, adj_in, adj_out] 就是一个list形式的tensor
        # fin_state = tf.nn.embedding_lookup()
        fin_state = inputs[0]
        adj_in = inputs[1]
        adj_out = inputs[2]
        # 似乎不需要循环？
        fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
        fin_state = tf.reshape(fin_state, [-1, self.out_size])
        # fin_state_in = tf.matmul(tf.reshape(fin_state, [-1, self.out_size]), self.W_in)
        fin_state_in = tf.matmul(fin_state, self.W_in) + self.b_in
        fin_state_in = tf.reshape(fin_state_in, [self.batch_size, -1, self.out_size])
        fin_state_out = tf.matmul(fin_state, self.W_out) + self.b_out
        fin_state_out = tf.reshape(fin_state_out, [self.batch_size, -1, self.out_size])
        av = tf.concat(values=[tf.matmul(adj_in, fin_state_in), tf.matmul(adj_out, fin_state_out)], axis=-1)
        av = tf.reshape(av, [-1, 2 * self.out_size])
        # RNN
        rnn_layer = tf.keras.layers.RNN(cell=self.cell,
                                        return_sequences=True,
                                        return_state=True)
        state_output, fin_state = rnn_layer(inputs=tf.expand_dims(av, axis=1),
                                            initial_state=fin_state)
        return tf.reshape(fin_state, [self.batch_size, -1, self.out_size])


class MyModel(tf.keras.Model):
    def __init__(self, n_node, hidden_size=100, out_size=100, batch_size=100):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.stdv = 1.0 / math.sqrt(self.hidden_size)
        self.l2 = keras.regularizers.L2(l2=1e-5)
        self.w1 = tf.Variable(tf.random.uniform((self.out_size, self.out_size), -self.stdv, self.stdv), name='w1',
                              dtype=tf.float32)
        self.w2 = tf.Variable(tf.random.uniform((self.out_size, self.out_size), -self.stdv, self.stdv), name='w2',
                              dtype=tf.float32)
        self.q = tf.Variable(tf.random.uniform(shape=(1, self.out_size), minval=-self.stdv, maxval=self.stdv), name='q',
                             dtype=tf.float32)
        self.c = tf.Variable(tf.zeros(shape=(self.out_size,)), name='c', dtype=tf.float32)
        self.w3 = tf.Variable(tf.random.uniform((2 * self.out_size, self.out_size), -self.stdv, self.stdv), name='w3',
                              dtype=tf.float32)
        self.embedding = tf.Variable(tf.random.uniform(shape=(self.n_node, self.hidden_size), minval=-self.stdv,
                                                       maxval=self.stdv), name='embedding', dtype=tf.float32)
        self.ggnn_layer = GGNN(batch_size=100,
                               hidden_size=self.hidden_size,
                               out_size=self.out_size)

    def call(self, inputs, training=None, mask=None):
        # [item, adj_in, adj_out, mask, alias, labels] in a tensor

        adj_in = inputs[0]
        adj_out = inputs[1]
        alias = inputs[2]
        # alias = tf.cast(alias, dtype=tf.int32)
        item = inputs[3]
        # item = tf.cast(item, dtype=tf.int32)
        mask = inputs[4]

        fin_state = tf.nn.embedding_lookup(self.embedding, item)
        # 完成定义变量的类型转换  ----》 tf.float32
        adj_in = tf.cast(adj_in, tf.float32)
        adj_out = tf.cast(adj_out, tf.float32)
        mask = tf.cast(mask, tf.float32)

        re_embedding = self.ggnn_layer([fin_state, adj_in, adj_out])
        rm = tf.reduce_sum(mask, 1)

        #  indices 中 表示索引的 部分 被提取到的值替换
        last_id = tf.gather_nd(params=alias,
                               indices=tf.stack([tf.range(self.batch_size), tf.cast(rm, tf.int32) - 1], axis=1))
        vertex_n = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))
        session_vertexes_i = tf.stack(
            [tf.nn.embedding_lookup(re_embedding[i], alias[i]) for i in range(self.batch_size)],
            axis=0)
        W1_Vn = tf.matmul(vertex_n, self.w1)
        W2_Vi = tf.matmul(tf.reshape(session_vertexes_i, [-1, self.out_size]), self.w2)
        W1_Vn = tf.reshape(W1_Vn, [self.batch_size, 1, -1])
        soft_attention_sigmoid = tf.keras.activations.sigmoid(
            W1_Vn + tf.reshape(W2_Vi, [self.batch_size, -1, self.out_size]) + self.c)
        alpha = (tf.matmul(tf.reshape(soft_attention_sigmoid, [-1, self.out_size]), self.q, transpose_b=True) *
                 tf.reshape(mask, [-1, 1]))
        # V whether use the whole vertex [0:]  or [1:]
        V = self.embedding[1:]
        # the local embedding can be simply defined as Vn of the last-clicked item Vs,n, i.e. Sl = Vn.
        # here is the different W1_Vn or Vn
        session_global = tf.reduce_sum(tf.reshape(alpha, [self.batch_size, -1, 1]) * session_vertexes_i, 1)
        # W1_Vn
        # session_hidden = tf.matmul(tf.concat([session_global, tf.reshape(W1_Vn, [-1, self.out_size])], -1), self.w3)
        # Vn
        session_hidden = tf.matmul(tf.concat([session_global, vertex_n], -1), self.w3)
        z_score = tf.matmul(session_hidden, V, transpose_b=True)
        y = tf.keras.activations.softmax(z_score)

        return y
