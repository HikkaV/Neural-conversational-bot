import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--path_decoder',
                        help='Path to decoder model.',
                        type=str)
    parser.add_argument('-e', '--path_encoder',
                        help='Path to encoder model.',
                        type=str)
    parser.add_argument('-p', '--path_mapping',
                        help='Path to token mapping used in the model.',
                        type=str)
    parser.add_argument('-m', '--max_len',
                        help='Max len of the sequence used while model training',
                        type=int,
                        default=10)
    return parser.parse_args()
