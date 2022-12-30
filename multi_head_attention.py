import tensorflow as tf
from keras import layers


class ScaledDotProductAttention(layers.Layer):
    """
        softmax(Q*K.T/sqrt(d_k))*V
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def call(self, querys, keys, values, d_k, mask=None):
        print("ScaledDotProductAttention: q: {}, k: {}, v: {}, d_k:{}"
              .format(querys.shape, keys.shape, values.shape, d_k))

        # Q*K.T/sqrt(d_k)
        scores = tf.matmul(querys, keys, transpose_b=True) / tf.math.sqrt(tf.cast(d_k, tf.float32))

        # 使用mask乘以一个很小的数，使得mask的部分经过softmax接近0
        if mask is not None:
            scores += -1e9 * mask

        weights = tf.nn.softmax(scores)

        result = tf.matmul(weights, values)

        print("ScaledDotProductAttention: result: {}".format(result.shape))

        return result


class MultiHeadAttention(layers.Layer):

    def __init__(self, headers, d_k, d_v, d_out):
        super(MultiHeadAttention, self).__init__()

        # Number of attention heads to use
        self.headers = headers
        # Dimensionality of the linearly projected queries and keys
        self.d_k = d_k
        # Dimensionality of the linearly projected values
        self.d_v = d_v
        # Dimensionality of the model output
        self.d_out = d_out

        # Scaled dot product attention
        self.attention = ScaledDotProductAttention()

        # Learned projection matrix for the queries
        self.W_q = layers.Dense(d_k)
        # Learned projection matrix for the keys
        self.W_k = layers.Dense(d_k)
        # Learned projection matrix for the values
        self.W_v = layers.Dense(d_v)
        # Learned projection matrix for the multi-head output
        self.W_o = layers.Dense(d_out)

    def reshape(self, x, headers, flag):
        if flag:
            # x_shape: [batch, seq_length, d_k or d_v]
            # reshape to: [batch, seq_length, headers, -1]
            x = tf.reshape(x, shape=[x.shape[0], x.shape[1], headers, -1])
            # Tensor shape after reshaping and transposing: [batch_size, heads, seq_length, -1]
            x = tf.transpose(x, perm=[0, 2, 1, 3])
        else:
            # x_shape: [batch_size, heads, seq_length, -1]
            # transpose to: [batch, seq_length, heads, -1]
            x = tf.transpose(x, perm=[0, 2, 1, 3])
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_v)
            x = tf.reshape(x, shape=[x.shape[0], x.shape[1], self.d_v])

        return x

    def call(self, querys, keys, values, mask=None):
        # input:
        # q: (64, 5, 64)
        # k: (64, 5, 64)
        # v: (64, 5, 64)
        print("0. q: {}, k: {}, v: {}".format(querys.shape, keys.shape, values.shape))

        # q: (64, 5, 64)
        # k: (64, 5, 64)
        # v: (64, 5, 64)
        q = self.W_q(querys)
        k = self.W_k(keys)
        v = self.W_v(values)

        print("1. q: {}, k: {}, v: {}".format(q.shape, k.shape, v.shape))

        # [batch_size, heads, input_seq_length, -1]
        # q: (64, 8, 5, 8)
        # k: (64, 8, 5, 8)
        # v: (64, 8, 5, 8)
        q = self.reshape(q, self.headers, True)
        k = self.reshape(k, self.headers, True)
        v = self.reshape(v, self.headers, True)

        print("2. q: {}, k: {}, v: {}".format(q.shape, k.shape, v.shape))

        # [batch_size, heads, input_seq_length, -1]
        # (64, 8, 5, 8)
        output = self.attention(q, k, v, self.d_k, mask)

        print("3. output: {}".format(output.shape))

        # [batch_size, input_seq_length, d_v]
        # (64, 5, 64)
        output = self.reshape(output, self.headers, False)

        print("4. output: {}".format(output.shape))

        # [batch_size, input_seq_length, d_model]
        # (64, 5, 512)
        output = self.W_o(output)
        print("5. output: {}".format(output.shape))

        return output


from numpy import random

input_seq_length = 5  # Maximum length of the input sequence
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of the model sub-layers' outputs
batch_size = 64  # Batch size from the training process

queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
multihead_attention(queries, keys, values)
