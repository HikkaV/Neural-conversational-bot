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
        self.dict_params = {"beam": {"beam_size": 7},
                            "nucleus": {"top_p": 0.95, "temperature": 1},
                            }
        self.setting_message = False
        self.setting_to_change = None
        self.decoding_type = "greedy"

        @self.BOT.callback_query_handler(func=lambda call: True)
        def callback_worker(call):
            if call.data in ['beam', 'nucleus', 'greedy']:
                if call.data == 'beam':
                    self.predict = predict_beam
                    self.decoding_strategy = self.beam_decoder
                elif call.data == 'nucleus':
                    self.predict = predict_nucleus
                    self.decoding_strategy = self.nucleus_decoder
                elif call.data == 'greedy':
                    self.predict = predict_greedy
                    self.decoding_strategy = self.greedy_decoder

                self.decoding_type = call.data
            else:
                self.setting_to_change = call.data

        @self.BOT.message_handler(content_types=['text'],
                                  )
        def reply(message):
            if message.text == "/help":
                self.BOT.send_message(message.from_user.id,
                                      "Hello. In order to start the bot please write /start.")
            elif message.text == "/start" or message.text == "/choose":
                self.BOT.send_message(message.from_user.id,
                                      "Please choose type of decoding and then start the conversation.",
                                      reply_markup=self.keys_decoding)
            elif message.text == "/show_default":
                if self.decoding_type != 'greedy':
                    self.BOT.send_message(message.from_user.id,
                                          "Default settings for used decoding strategy ({0}) are the following : {1}" \
                                          .format(self.decoding_type, self.dict_params[self.decoding_type]))
                else:
                    self.BOT.send_message(message.from_user.id, "No settings for greedy decoding")
            elif message.text == '/change_default':
                if self.decoding_type != 'greedy':
                    keyboard = self.set_keys_options()
                    self.BOT.send_message(message.from_user.id,
                                          "Please choose the parameter to change and write the new value.",
                                          reply_markup=keyboard)
                    self.setting_message = True
                else:
                    self.BOT.send_message(message.from_user.id, "No settings for greedy decoding.")

            elif self.setting_message:
                try:
                    param = float(message.text)
                except Exception as e:
                    param = None

                if param:
                    initial_param = param
                    param = max(param, 0)
                    if self.decoding_type == 'nucleus':
                        param = min(param, 1.0)
                        param = max(param, 0.1)
                    else:
                        param = max(int(param), 2)
                    self.dict_params[self.decoding_type][self.setting_to_change] = param
                    if initial_param != param:
                        self.BOT.send_message(message.from_user.id, "Successfully set value of {0} "
                                                                    "for param {1}. "
                                                                    "Initial param passed by user ({1}={2}) "
                                                                    "was changed to fir constraints. ".format(param,
                                                                                                              self.setting_to_change,
                                                                                                              initial_param))
                    else:
                        self.BOT.send_message(message.from_user.id, "Successfully set value of {0} "
                                                                    "for param {1}".format(param,
                                                                                           self.setting_to_change))
                else:
                    self.BOT.send_message(message.from_user.id, "The error occurred, please try again with "
                                                                "command /change_default. ")

                self.setting_message = False
            else:
                processed_sentence = process_sentence(message.text, self.token_mapping,
                                                      max_len=self.max_len)
                if self.decoding_type == 'beam':
                    result = self.predict(self.decoding_strategy, processed_sentence,
                                          self.inverse_token_mapping,
                                          beam_size=self.dict_params[self.decoding_type]['beam_size'])
                elif self.decoding_type == 'nucleus':
                    result = self.predict(self.decoding_strategy, processed_sentence,
                                          self.inverse_token_mapping,
                                          top_p=self.dict_params[self.decoding_type]['top_p'],
                                          temperature=self.dict_params[self.decoding_type]['temperature'])
                else:
                    result = self.predict(self.decoding_strategy, processed_sentence,
                                          self.inverse_token_mapping)

                self.BOT.send_message(message.from_user.id, result)

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

    def set_keys_options(self):
        keyboard = types.InlineKeyboardMarkup()
        possible_keys = self.dict_params[self.decoding_type]
        for key in possible_keys.keys():
            keyboard.add(types.InlineKeyboardButton(text=key,
                                                    callback_data=key))
        return keyboard

    def start(self):
        self.BOT.polling(none_stop=True, interval=0)
