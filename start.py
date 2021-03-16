from telegram_bot.bot import Seq2Seq, TelegramBot
from utils.load_utils import load_json

PATH = "token_mapping_cornell.json"
mapping = load_json(PATH)
inverse_mapping = dict((v,k) for k, v in mapping.items())
pad_token = inverse_mapping[0]
start_token = inverse_mapping[1]
end_token = inverse_mapping[2]
unk_token = inverse_mapping[3]
model =  Seq2Seq(mapping,
         pad_token=mapping[pad_token],
         end_token=mapping[end_token],
         start_token=mapping[start_token],
         max_len=10,
         path_decoder="decoder_all_data.h5",
         path_encoder="encoder_all_data.h5")

bot = TelegramBot(model, mapping,
            end_token=mapping[end_token],
            start_token=mapping[start_token],
            max_len=10)
bot.start()