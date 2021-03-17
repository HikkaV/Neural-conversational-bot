from telegram_bot.bot import Seq2Seq, TelegramBot
from utils.load_utils import load_json
from utils.argparse_utils import parse_args

if __name__ == '__main__':
    args = parse_args()
    # loading mapping
    mapping = load_json(args.path_mapping)
    # taking out all the needed tokens
    inverse_mapping = dict((v, k) for k, v in mapping.items())
    pad_token = inverse_mapping[0]
    start_token = inverse_mapping[1]
    end_token = inverse_mapping[2]
    unk_token = inverse_mapping[3]
    # initialization of model
    model = Seq2Seq(mapping,
                    pad_token=mapping[pad_token],
                    end_token=mapping[end_token],
                    start_token=mapping[start_token],
                    max_len=args.max_len,
                    path_decoder=args.path_decoder,
                    path_encoder=args.path_encoder)
    # init bot
    bot = TelegramBot(model, mapping,
                      end_token=mapping[end_token],
                      start_token=mapping[start_token],
                      max_len=args.max_len)
    # start it
    bot.start()
