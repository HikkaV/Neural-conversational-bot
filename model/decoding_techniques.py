import tensorflow as tf
import numpy as np


class Decoder:
    def __init__(self, encoder: tf.keras.Model,
                 decoder: tf.keras.Model, **kwargs):
        self.encoder = encoder
        self.decoder = decoder

    def decode(self, input):
        pass


class GreedyDecoder(Decoder):
    def __init__(self, encoder: tf.keras.Model,
                 decoder: tf.keras.Model,
                 start_token: int,
                 end_token: int,
                 max_len: int = 10,
                 **kwargs):
        super().__init__(encoder, decoder, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.max_len = max_len
        self.start_token = start_token
        self.end_token = end_token

    def decode(self, input, max_len_output=50, return_attention=False):
        input = tf.keras.preprocessing.sequence.pad_sequences([input], padding='post')
        encoder_hidden_states, state_h, state_c = self.encoder(input)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.start_token

        attention_plot = []
        res = []
        while True:
            # Sample a token
            output, state_h, state_c, attention_weights = self.decoder([target_seq,
                                                                        state_h,
                                                                        state_c,
                                                                        encoder_hidden_states])
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot.append(attention_weights.numpy())
            sampled_token_index = np.argmax(output)

            if len(res) > max_len_output or sampled_token_index == self.end_token:
                break

            res.append(sampled_token_index)

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

        attention_plot = np.array(attention_plot)
        if return_attention:
            return res, attention_plot
        else:
            return res
