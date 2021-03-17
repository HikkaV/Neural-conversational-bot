import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils.processing_utils import uncover_reduction, clean_bad_chars


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def map_to_ids(x, mapping, end_token=None, start_token=None, padding_token=None, max_len=None,
               return_len=False):
    if isinstance(x, str):
        x = x.split(' ')
    max_len = max_len if max_len else len(x)
    length = len(x)
    if start_token:
        x = [start_token] + x
    sent_ids = [mapping[word] for word in x[:max_len]]
    if max_len > len(x):
        if end_token:
            sent_ids.append(mapping[end_token])
        if padding_token:
            sent_ids += [mapping[padding_token]] * (max_len - len(sent_ids))
        if return_len:
            return sent_ids, length + 1
        else:
            return sent_ids
    if end_token:
        sent_ids[-1] = mapping[end_token]
    if return_len:
        return sent_ids, max_len
    else:
        return sent_ids


def process(x, token_mapping, unk_token):
    res = []
    for i in x.split(' '):
        if token_mapping.get(i):
            res.append(i)
        elif token_mapping.get(i.lower()):
            res.append(i.lower())
        else:
            res.append(unk_token)
    return res


def process_sentence(sentence,
                token_mapping,
                pad_unk=True,
                process_sentence=True,
                max_len=10,
                ):
    if process_sentence:
        cleaned_sentence = uncover_reduction(clean_bad_chars(sentence))
        unk_token = "<pad>" if pad_unk else "<unk>"
        processed_sentence = process(cleaned_sentence, token_mapping, unk_token)
        mapped_sentence = map_to_ids(processed_sentence, token_mapping, "<end>", padding_token="<pad>",
                                     max_len=max_len)
    else:
        mapped_sentence = sentence

    return mapped_sentence


def predict_beam(decoder, mapped_sentence, inverse_token_mapping, beam_size=7):
    prediction = decoder.decode(mapped_sentence, beam_size)
    answer = " ".join([inverse_token_mapping.get(i) for i in prediction]).capitalize()
    return answer


def predict_nucleus(decoder, mapped_sentence, inverse_token_mapping, len_output=50, top_p=0.75):
    prediction = decoder.decode(mapped_sentence, len_output, top_p)
    answer = " ".join([inverse_token_mapping.get(i) for i in prediction]).capitalize()
    return answer


def predict_greedy(decoder, mapped_sentence, inverse_token_mapping, len_output=50):
    prediction = decoder.decode(mapped_sentence, len_output)
    answer = " ".join([inverse_token_mapping.get(i) for i in prediction]).capitalize()
    return answer
