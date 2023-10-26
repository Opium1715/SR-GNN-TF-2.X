import tensorflow as tf
import math


class Model(object):
    def __init__(self, n_node, l2, step, lr, decay, lr_dc, hidden_size=100, out_size=100, batch_size=100, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.L2 = l2
        self.n_node = n_node
        self.step = step

        self.stdv = 1.0 / math.sqrt(self.hidden_size)
        # 后续计算 --------------------------------------------------------------------------------------------- #
        self.w1 = tf.Variable(tf.random.uniform((self.out_size, self.out_size), -self.stdv, self.stdv), name='w1',
                              dtype=tf.float32)
        self.w2 = tf.Variable(tf.random.uniform((self.out_size, self.out_size), -self.stdv, self.stdv), name='w2',
                              dtype=tf.float32)
        self.v = tf.Variable(tf.random.uniform(shape=(1, self.out_size), minval=-self.stdv, maxval=self.stdv), name='v',
                             dtype=tf.float32)
        self.b = tf.Variable(tf.zeros(shape=(self.out_size,)), name='b', dtype=tf.float32)
        self.embedding = tf.Variable(tf.random.uniform(shape=(self.n_node, self.hidden_size), minval=-self.stdv,
                                                       maxval=self.stdv), name='embedding', dtype=tf.float32)
        # GGNN ------------------------------------------------------------------------------------------------ #
        self.W_in = tf.Variable(tf.random.uniform((self.out_size, self.out_size), -self.stdv, self.stdv), name='W_in',
                                dtype=tf.float32)
        self.b_in = tf.Variable(tf.random.uniform((self.out_size,), -self.stdv, self.stdv), name='b_in',
                                dtype=tf.float32)
        self.W_out = tf.Variable(tf.random.uniform((self.out_size, self.out_size), -self.stdv, self.stdv), name='W_out',
                                 dtype=tf.float32)
        self.b_out = tf.Variable(tf.random.uniform((self.out_size,), -self.stdv, self.stdv), name='b_out',
                                 dtype=tf.float32)
        self.B = tf.Variable(tf.random.uniform((2 * self.out_size, self.out_size), -self.stdv, self.stdv), name='B',
                             dtype=tf.float32)
        # ------------------------------------------------------------------------------------------------------ #
        self.learning_rate = tf.optimizers.schedules.ExponentialDecay(lr, decay, decay_rate=lr_dc, staircase=True)
        self.opt = tf.optimizers.Adam(self.learning_rate)

    def train_step(self, item, adj_in, adj_out, mask, alias, labels, train=True):
        self.batch_size = tf.shape(item)[0]
        variables = [self.w1, self.w2, self.b, self.v, self.W_in, self.b_in, self.W_out, self.b_out, self.B,
                     self.embedding]
        with tf.GradientTape() as tape:
            loss, logits = self.forward(item, adj_in, adj_out, mask, alias, labels, train)
            grads = tape.gradient(loss, variables)
            if train:
                self.opt.apply_gradients(zip(grads, variables))

        return loss, logits

    def forward(self, item, adj_in, adj_out, mask, alias, labels, train):
        fin_state = tf.nn.embedding_lookup(self.embedding, item)
        cell = tf.keras.layers.GRUCell(self.out_size)

        # 完成定义变量的类型转换  ----》 tf.float32
        adj_in = tf.cast(adj_in, dtype=tf.float32)
        adj_out = tf.cast(adj_out, tf.float32)
        mask = tf.cast(mask, tf.float32)

        for i in range(self.step):
            fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
            # item * W_in + b_in
            fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]), self.W_in) + self.b_in,
                                      [self.batch_size, -1, self.out_size])
            fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]), self.W_out) + self.b_out,
                                       [self.batch_size, -1, self.out_size])
            av = tf.concat([tf.matmul(adj_in, fin_state_in), tf.matmul(adj_out, fin_state_out)], axis=-1)
            # 这里似乎可以改用2.0写法，稍后试试
            # state_output, fin_state = tf.compat.v1.nn.dynamic_rnn(cell=cell, inputs=tf.expand_dims( tf.reshape(av,
            # [-1, 2 * self.out_size]), axis=1), initial_state=tf.reshape(fin_state, [-1, self.out_size]))
            # tf2.x写法
            rnn_layer = tf.keras.layers.RNN(cell=cell, return_state=True, return_sequences=True)
            state_output, fin_state = rnn_layer(tf.expand_dims(tf.reshape(av, [-1, 2 * self.out_size]), axis=1),
                                                initial_state=tf.reshape(fin_state, [-1, self.out_size]))
            re_embedding = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
            # s
            rm = tf.reduce_sum(mask, 1)
            last_id = tf.gather_nd(alias, tf.stack([tf.range(self.batch_size), tf.cast(rm, tf.int32) - 1], axis=1))
            last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))
            seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], alias[i]) for i in range(self.batch_size)],
                             axis=0)
            last = tf.matmul(last_h, self.w1)
            seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.w2)
            last = tf.reshape(last, [self.batch_size, 1, -1])
            m = tf.nn.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, self.out_size]) + self.b)
            coef = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.v, transpose_b=True) * tf.reshape(mask,
                                                                                                        [-1, 1])
            b = self.embedding[1:]
            # ---------------------------------------------------------------------------------------------------------
            ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1),
                            tf.reshape(last, [-1, self.out_size])], -1)
            y1 = tf.matmul(ma, self.B)
            logits = tf.matmul(y1, b, transpose_b=True)
            softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels - 1, logits=logits)
            loss = tf.reduce_mean(softmax)

            if train:
                variables = [self.w1, self.w2, self.b, self.v, self.W_in, self.b_in, self.W_out,
                             self.b_out, self.B, self.embedding]
                l2_losses = [tf.nn.l2_loss(v) for v in variables]
                lossL2 = tf.add_n(l2_losses) * self.L2
                loss = loss + lossL2

            return loss, logits


# 结构化的写法
