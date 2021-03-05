from model.layers import BahdanauAttention, PartialEmbeddingsUpdate
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs\n",
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized\n",
        print(e)
tf.random.set_seed(5)
import numpy as np
from datetime import datetime
from utils.dir_utils import mkdir
from utils.dir_utils import os
import mlflow
import tqdm
from model.decoding_techniques import Decoder, GreedyDecoder


class Seq2Seq:
    def __init__(self,
                 token_mapping: dict,
                 pad_token: int,
                 start_token: int,
                 end_token: int,
                 max_len: int = 10,
                 embeddings: np.array = None,
                 embedding_prefix: str = None,
                 missing_tokens: np.array = None,
                 fine_tune: bool = False,
                 emb_units: int = 300,
                 lstm_units: int = 128,
                 attention_units: int = 128,
                 dropout_prob: float = 0.,
                 decoding_strategy: Decoder = GreedyDecoder,
                 path_decoder: str = "",
                 path_encoder: str = ""
                 ):
        self.embedding_prefix = embedding_prefix
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.pretrained_embs = not (embeddings is None)
        self.embeddings = embeddings
        self.missing_tokens = missing_tokens if not (missing_tokens is None) else np.array([])
        self.fine_tune = fine_tune
        self.emb_units = emb_units
        self.token_mapping = token_mapping
        self.inverse_token_mapping = dict((v, k) for k, v in token_mapping.items())
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dropout_prob = dropout_prob
        if not (path_encoder and path_decoder):
            self.encoder, self.decoder = self.__build_models()
        else:
            self.load_models(path_encoder, path_decoder)
        self.params = {"pretrained_embs": self.pretrained_embs,
                       "fine_tune": self.fine_tune,
                       "missing_tokens": len(self.missing_tokens),
                       "len_mapping": len(token_mapping),
                       "emb_units": emb_units,
                       "lstm_units": lstm_units,
                       "attention_units": attention_units,
                       "dropout_prob": dropout_prob,
                       "length_token_mapping": len(token_mapping)
                       }
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none',
                                                                         from_logits=True)
        self.predict = decoding_strategy(self.encoder,
                                         self.decoder,
                                         start_token,
                                         end_token,
                                         max_len).decode

    def load_models(self, encoder_path, decoder_path):
        self.encoder = tf.keras.models.load_model(
            encoder_path,
            custom_objects={"PartialEmbeddingsUpdate": PartialEmbeddingsUpdate})
        self.decoder = tf.keras.models.load_model(
            decoder_path,
            custom_objects={"PartialEmbeddingsUpdate": PartialEmbeddingsUpdate,
                            "BahdanauAttention": BahdanauAttention})

    def save_models(self, encoder_path, decoder_path):
        self.encoder.save(encoder_path)
        self.decoder.save(decoder_path)

    def __build_models(self):
        # encoder
        enc_input = tf.keras.Input(shape=(None,), name='enc_input')
        # shared embedding layer
        if not self.pretrained_embs:
            embedding_layer = tf.keras.layers.Embedding(len(self.token_mapping), self.emb_units, name='embedding')
        else:
            emb_units = self.embeddings.shape[1]
            if self.fine_tune:
                embedding_layer = tf.keras.layers.Embedding(len(self.token_mapping), emb_units,
                                                            weights=[self.embeddings],
                                                            name='embedding')
            elif self.missing_tokens.size == 0:
                embedding_layer = tf.keras.layers.Embedding(len(self.token_mapping), emb_units,
                                                            weights=[self.embeddings],
                                                            trainable=False,
                                                            name='embedding')
            else:
                embedding_layer = PartialEmbeddingsUpdate(len(self.token_mapping), emb_units, self.missing_tokens,
                                                          weights=[self.embeddings],
                                                          name='embeddings')

        embedded_enc = embedding_layer(enc_input)

        encoder = tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, return_state=True, name='encoder',
                                       dropout=self.dropout_prob
                                       )
        encoder_hidden_states, state_h, state_c = encoder(embedded_enc)

        # building encoder model
        encoder = tf.keras.Model(enc_input, [encoder_hidden_states, state_h, state_c])
        # decoder
        dec_input = tf.keras.Input(shape=(None,), name='dec_input')
        dec_state_input_h = tf.keras.Input(shape=(self.lstm_units,), name='state_h')
        dec_state_input_c = tf.keras.Input(shape=(self.lstm_units,), name='state_c')
        encoder_hidden_states = tf.keras.Input(shape=(None, self.lstm_units), name='enc_hidden_states')
        # applying attention
        attention = BahdanauAttention(self.attention_units)
        context_vector, attention_weights = attention(dec_state_input_h, encoder_hidden_states)
        embedded_dec = embedding_layer(dec_input)
        inp = tf.concat([tf.expand_dims(context_vector, 1), embedded_dec], axis=-1)
        decoder = tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, return_state=True, name='decoder',
                                       dropout=self.dropout_prob)
        # giving everything to output layer
        decoder_hidden_states, next_state_h, next_state_c = decoder(inp, initial_state=[dec_state_input_h,
                                                                                        dec_state_input_c])
        decoder_hidden_states = tf.reshape(decoder_hidden_states, (-1, decoder_hidden_states.shape[2]))
        output_layer = tf.keras.layers.Dense(len(self.token_mapping), activation=None, name='dense_logits')
        result = output_layer(decoder_hidden_states)
        decoder = tf.keras.Model([dec_input, dec_state_input_h, dec_state_input_c, encoder_hidden_states],
                                 [result, next_state_h, next_state_c, attention_weights])
        return encoder, decoder

    def fit(self, train_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset = None,
            learning_rate=1e-3,
            num_epochs: int = 100,
            batch_size: int = 256,
            steps_per_epoch: int = 1000,
            epochs_patience: int = 10,
            dir_save: str = 'models',
            experiment_name: str = "tmp"
            ):
        mlflow.set_experiment(experiment_name=experiment_name)
        mkdir(dir_save)
        now_date = datetime.strftime(datetime.now(), "%Y-%m-%d")
        template = "date:{}_fine_tune:{}_mode:{}".format(now_date, self.fine_tune, self.embedding_prefix)
        encoder_path = ""
        decoder_path = ""
        with mlflow.start_run(run_name=str(template)):

            train_dataset = train_dataset.shuffle(batch_size).batch(batch_size)
            validation_dataset = validation_dataset.batch(batch_size) if not (
                    validation_dataset is None) else validation_dataset
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            train_loss = []
            val_loss = []
            train_perplexity = []
            val_perplexity = []
            val_loss_bound = 0
            epochs_overfit = 0
            self.params.update({"epochs": num_epochs,
                                "batch_size": batch_size,
                                "steps_per_epoch": steps_per_epoch,
                                "epochs_patience": epochs_patience,
                                "learning_rate": learning_rate})
            mlflow.log_params(self.params)
            for epoch in tqdm.tqdm(range(num_epochs)):
                total_train_loss = []
                total_val_loss = []
                total_train_perplexity = []
                total_val_perplexity = []

                for batched_x_enc, batched_x_dec, batched_y, batched_length in train_dataset.take(steps_per_epoch):
                    batch_loss, batch_perplexity = self.train_step(np.array(batched_x_enc),
                                                                   np.array(batched_x_dec),
                                                                   np.array(batched_y),
                                                                   np.array(batched_length),
                                                                   optimizer)
                    total_train_loss.append(batch_loss)
                    total_train_perplexity.append(batch_perplexity)

                total_train_perplexity = np.mean(total_train_perplexity)
                total_train_loss = np.mean(total_train_loss)

                mlflow.log_metric('train_loss', total_train_loss, step=epoch)
                mlflow.log_metric('train_perplexity', total_train_perplexity, step=epoch)

                print('\n')
                print('Epoch {} train loss {:.4f} train perplexity {:.4f}'.format(epoch,
                                                                                  total_train_loss,
                                                                                  total_train_perplexity))
                if not (validation_dataset is None):
                    for batched_x_test_enc, batched_x_test_dec, batched_test_y, batched_test_length in validation_dataset.take(
                            steps_per_epoch):
                        batch_loss, batch_perplexity = self.evaluate(np.array(batched_x_test_enc),
                                                                     np.array(batched_x_test_dec),
                                                                     np.array(batched_test_y),
                                                                     np.array(batched_test_length))

                        total_val_loss.append(batch_loss)
                        total_val_perplexity.append(batch_perplexity)

                    total_val_perplexity = np.mean(total_val_perplexity)
                    total_val_loss = np.mean(total_val_loss)

                    mlflow.log_metric('val_loss', total_val_loss, step=epoch)
                    mlflow.log_metric('val_perplexity', total_val_perplexity, step=epoch)

                    if epoch == 0:
                        val_loss_bound = total_val_loss

                    if total_val_loss > val_loss_bound:
                        epochs_overfit += 1
                        if epochs_overfit == 1:
                            encoder_path = os.path.join(dir_save,
                                                        'encoder_{}_epoch:{}.h5'.format(template, epoch))
                            decoder_path = os.path.join(dir_save,
                                                        'decoder_{}_epoch:{}.h5'.format(template, epoch))
                            self.save_models(encoder_path=encoder_path,
                                             decoder_path=decoder_path)
                    else:
                        val_loss_bound = total_val_loss
                        epochs_overfit = 0

                    print('\n')
                    print('Epoch {} validation loss {:.4f} validation perplexity {:.4f}'.format(epoch,
                                                                                                total_val_loss,
                                                                                                total_val_perplexity))
                    print('\n')
                    val_loss.append(total_val_loss)
                    val_perplexity.append(total_val_perplexity)

                train_loss.append(total_train_loss)
                train_perplexity.append(total_train_perplexity)

                if epochs_overfit == epochs_patience:
                    print('Validation loss has not improved for last {} epochs, stopping training!'.format(
                        epochs_patience))
                    break

            if encoder_path and decoder_path:
                self.load_models(encoder_path, decoder_path)
            print('Finished training')

    @tf.function
    def evaluate(self, encoder_input, decoder_input, target, lengths):
        enc_hidden_states, state_h, state_c = self.encoder(encoder_input)
        # Teacher forcing - feeding the target as the next input
        batched_loss = []
        for t in range(decoder_input.shape[1]):
            # passing enc_output to the decoder
            dec_input = tf.expand_dims(decoder_input[:, t], 1)
            result, state_h, state_c, attention_weights = self.decoder([dec_input,
                                                                        state_h,
                                                                        state_c,
                                                                        enc_hidden_states])
            batched_loss.append(self.loss_function(target[:, t], result))

        batched_loss = tf.reshape(tf.stack(batched_loss), shape=decoder_input.shape)
        batched_loss = tf.reduce_sum(batched_loss, axis=1)
        lengths = tf.cast(lengths, dtype=batched_loss.dtype)
        loss = tf.reduce_mean(batched_loss / lengths)

        perplexity = tf.exp(loss)

        return loss, perplexity

    @tf.function
    def train_step(self, encoder_input, decoder_input, target, lengths, optimizer):

        with tf.GradientTape() as tape:
            enc_hidden_states, state_h, state_c = self.encoder(encoder_input)
            # Teacher forcing - feeding the target as the next input
            batched_loss = []
            for t in range(decoder_input.shape[1]):
                # passing enc_output to the decoder
                dec_input = tf.expand_dims(decoder_input[:, t], 1)
                result, state_h, state_c, attention_weights = self.decoder([dec_input,
                                                                            state_h,
                                                                            state_c,
                                                                            enc_hidden_states])

                batched_loss.append(self.loss_function(target[:, t], result))
            batched_loss = tf.reshape(tf.stack(batched_loss), shape=decoder_input.shape)
            batched_loss = tf.reduce_sum(batched_loss, axis=1)
            lengths = tf.cast(lengths, dtype=batched_loss.dtype)
            loss = tf.reduce_mean(batched_loss / lengths)

        perplexity = tf.exp(loss)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return loss, perplexity

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, self.pad_token))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return loss_
