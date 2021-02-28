import tensorflow as tf


class PartialEmbeddingsUpdate(tf.keras.layers.Layer):
    def __init__(self, input_dim,
                 output_dim,
                 indices_to_update,
                 initializer='glorot_uniform',
                 **kwargs):
        super(PartialEmbeddingsUpdate, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initializer = initializer
        self.indices_to_update = indices_to_update

    def build(self, input_shape):
        self.embeddings = self.add_weight(shape=(self.input_dim, self.output_dim),
                                          initializer=self.initializer,
                                          trainable=True)
        self.bool_mask = tf.equal(tf.expand_dims(tf.range(0, self.input_dim), 1),
                                  tf.expand_dims(self.indices_to_update, 0))
        self.bool_mask = tf.reduce_any(self.bool_mask, 1)
        self.bool_mask_not = tf.logical_not(self.bool_mask)
        self.bool_mask_not = tf.expand_dims(tf.cast(self.bool_mask_not, dtype=self.embeddings.dtype), 1)
        self.bool_mask = tf.expand_dims(tf.cast(self.bool_mask, dtype=self.embeddings.dtype), 1)

    def call(self, input, **kwargs):
        input = tf.cast(input, dtype=tf.int32)
        embeddings = tf.stop_gradient(self.bool_mask_not * self.embeddings) + self.bool_mask * self.embeddings
        return tf.gather(embeddings, input)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'indices_to_update': self.indices_to_update,
            'initializer': self.initializer
        })
        return config


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(self.units)
        self.W2 = tf.keras.layers.Dense(self.units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values, **kwargs):
        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
        })
        return config
