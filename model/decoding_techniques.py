import tensorflow as tf
import numpy as np


class Decoder:
    def __init__(self, encoder: tf.keras.Model,
                 decoder: tf.keras.Model,
                 start_token: int,
                 end_token: int,
                 max_len: int = 10, ):
        self.encoder = encoder
        self.decoder = decoder
        self.max_len = max_len
        self.start_token = start_token
        self.end_token = end_token

    def decode(self, input):
        pass


class GreedyDecoder(Decoder):
    def __init__(self, encoder: tf.keras.Model,
                 decoder: tf.keras.Model,
                 start_token: int,
                 end_token: int,
                 max_len: int = 10,
                 ):
        super().__init__(encoder,
                         decoder,
                         start_token,
                         end_token,
                         max_len
                         )

    def decode(self, input, max_len_output=50, return_attention=False):
        input = tf.keras.preprocessing.sequence.pad_sequences([input], padding='post', maxlen=self.max_len)
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


class BeamSearchDecoder(Decoder):
    def __init__(self, encoder: tf.keras.Model,
                 decoder: tf.keras.Model,
                 start_token: int,
                 end_token: int,
                 max_len: int = 10,
                 ):
        super().__init__(encoder,
                         decoder,
                         start_token,
                         end_token,
                         max_len
                         )

    def decode(self, input, beam_size=3):
        start = [self.start_token]
        input = tf.keras.preprocessing.sequence.pad_sequences([input], padding='post', maxlen=self.max_len)

        encoder_hidden_states, state_h, state_c = self.encoder(input)

        start_word = [[start, 0.0, state_h, state_c]]

        while len(start_word[0][0]) < self.max_len:
            temp = []
            for s in start_word:
                target_seq = np.array([[s[0][-1]]])
                state_h = s[2]
                state_c = s[3]
                output, state_h, state_c, attention_weights = self.decoder([target_seq,
                                                                            state_h,
                                                                            state_c,
                                                                            encoder_hidden_states])
                output = np.hstack(output)
                output = tf.nn.softmax(output).numpy()
                word_preds = np.argsort(output)[-beam_size:]

                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += output[w]
                    temp.append([next_cap, prob, state_h, state_c, ])

            start_word = temp
            # Sorting according to the probabilities
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word = start_word[-beam_size:]

        start_word = start_word[-1][0]

        final_caption = []

        for i in start_word:
            if i != self.end_token:
                final_caption.append(i)
            else:
                break
        return final_caption[1:]

class NucleusDecoder(Decoder):
    def __init__(self, encoder: tf.keras.Model,
                 decoder: tf.keras.Model,
                 start_token: int,
                 end_token: int,
                 max_len: int = 10,
                 ):
        super().__init__(encoder,
                         decoder,
                         start_token,
                         end_token,
                         max_len
                         )

    def decode(self, input, max_len_output=50, top_p=0.95,
               temperature=1):
        input = tf.keras.preprocessing.sequence.pad_sequences([input], padding='post', maxlen=self.max_len)
        encoder_hidden_states, state_h, state_c = self.encoder(input)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.start_token

        res = []
        while True:
            # Sample a token
            output, state_h, state_c, attention_weights = self.decoder([target_seq,
                                                                        state_h,
                                                                        state_c,
                                                                        encoder_hidden_states])
            output = np.hstack(output)/temperature
            output = tf.nn.softmax(output).numpy()
            sorted_args = np.argsort(output)[::-1]
            sorted_probs = output[sorted_args]
            mask = np.cumsum(sorted_probs)<top_p
            if not mask.any():
                break
            sampled_token_index = np.random.choice(sorted_args[mask],p=tf.nn.softmax(sorted_probs[mask]).numpy())

            if len(res) > max_len_output or sampled_token_index == self.end_token:
                break

            res.append(sampled_token_index)

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

        return res