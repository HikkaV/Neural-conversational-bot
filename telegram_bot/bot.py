from model.seq2seq import Seq2Seq
from model.decoding_techniques import BeamSearchDecoder, GreedyDecoder, NucleusDecoder
from utils.model_utils import predict_beam, predict_greedy, predict_nucleus, process_sentence
import telebot
from telebot import types
import logging

logger = telebot.logger
telebot.logger.setLevel(logging.DEBUG)


class TelegramBot:
    BOT = telebot.TeleBot("1716865383:AAGR9GxP_cOefogxRgqN-b1NbXAQcgt-GDE")

    def __init__(self, model: Seq2Seq,
                 token_mapping: dict,
                 start_token: int,
                 end_token: int,
                 max_len: int = 10
                 ):

        self.decoder = model.decoder
        self.encoder = model.encoder
        self.token_mapping = token_mapping
        self.inverse_token_mapping = dict((v, k) for k, v in token_mapping.items())
        self.max_len = max_len
        self.greedy_decoder = GreedyDecoder(self.encoder,
                                            self.decoder,
                                            start_token,
                                            end_token,
                                            max_len)

        self.beam_decoder = BeamSearchDecoder(self.encoder,
                                              self.decoder,
                                              start_token,
                                              end_token,
                                              max_len)

        self.nucleus_decoder = NucleusDecoder(self.encoder,
                                              self.decoder,
                                              start_token,
                                              end_token,
                                              max_len)

        self.keys_decoding = self.set_keys_decoding()
        self.predict = predict_greedy
        self.decoding_strategy = self.greedy_decoder

    def set_keys_decoding(self):
        keyboard = types.InlineKeyboardMarkup()

        key_beam = types.InlineKeyboardButton(text='Beam search decoder',
                                              callback_data='beam')

        keyboard.add(key_beam)

        key_nucleus = types.InlineKeyboardButton(text='Nucleus decoder',
                                                 callback_data='nucleus')

        keyboard.add(key_nucleus)

        key_greedy = types.InlineKeyboardButton(text='Greedy decoder',
                                                callback_data='greedy')
        keyboard.add(key_greedy)

        return keyboard

    @BOT.message_handler(content_types=['text'], commands=['help'])
    def handle_help(self, message):
        self.BOT.send_message(message.from_user.id,
                              "Hello. In order to start the bot please write /start.")

    @BOT.message_handler(content_types=['text'], commands=['start'])
    def handle_start(self, message):
        self.BOT.send_message(message.from_user.id,
                              "Please choose type of decoding and then start the conversation.",
                              reply_markup=self.keys_decoding)

    @BOT.callback_query_handler(func=lambda call: True)
    def callback_worker(self, call):
        if call.data == 'beam':
            self.predict = predict_beam
            self.decoding_strategy = self.beam_decoder
        elif call.data == 'nucleus':
            self.predict = predict_nucleus
            self.decoding_strategy = self.nucleus_decoder

    @BOT.message_handler(content_types=['text'])
    def reply(self, message):
        processed_sentence = process_sentence(message, self.token_mapping,
                                              max_len=self.max_len)
        result = self.predict(self.decoding_strategy, processed_sentence,
                     self.inverse_token_mapping)
        self.BOT.send_message(message.from_user.id, result)

    def start(self):
        self.BOT.polling(none_stop=True, interval=0)
