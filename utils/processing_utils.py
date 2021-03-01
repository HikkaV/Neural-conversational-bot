import pandas as pd
import re

def read_chameleons(path, columns=None):
  with open(path, 'rb') as f:
    res = f.readlines()
  values = []
  for line in res:
    try:
      vals = line.decode().split(" +++$+++ ")
    except:
      vals = line.decode('cp1251').split(" +++$+++ ")
    values.append([i.strip() for i in vals])
  values = pd.DataFrame(data=values, columns=columns)
  return values

def cut_punctuation(x):
    punctuation = ['!', '?', '.']
    stop_crit = False
    new_sentence = []
    for char in x:
        if not (char in punctuation):
            stop_crit=False
            new_sentence.append(char)
        elif char in punctuation and not stop_crit:
            stop_crit=True
            new_sentence.append(char)
    new_sentence = ''.join(new_sentence)
    return new_sentence


def clean_bad_chars(x):
  x = re.sub(r"[^a-zA-Z(!?.)']+", " ", x).strip()
  x = ''.join(" "+i+" " if i in ['!', '?', '.'] else i for i in x).strip()
  return x


def get_upper_tokens(x):
    res = []
    if x:
        if ' ' in x:
            res = [i for i in x.split(' ') if i and i[0].isupper()]
        else:
            if x[0].isupper():
                res = [x]

    return res


def to_lower_case(x, to_lower: dict):
    res = ''
    if x:
        if ' ' in x:
            res = ' '.join(i.lower() if to_lower.get(i) else i for i in x.split(' '))
        else:
            if to_lower.get(x):
                res = x.lower()
    return res.strip()

def uncover_reduction(x):
  x = re.sub(r"i'm", "i am", x)
  x = re.sub(r"he's", "he is", x)
  x = re.sub(r"she's", "she is", x)
  x = re.sub(r"it's", "it is", x)
  x = re.sub(r"that's", "that is", x)
  x = re.sub(r"what's", "that is", x)
  x = re.sub(r"where's", "where is", x)
  x = re.sub(r"how's", "how is", x)
  x = re.sub(r"\'ll", " will", x)
  x = re.sub(r"\'ve", " have", x)
  x = re.sub(r"\'re", " are", x)
  x = re.sub(r"\'d", " would", x)
  x = re.sub(r"\'re", " are", x)
  x = re.sub(r"won't", "will not", x)
  x = re.sub(r"can't", "cannot", x)
  x = re.sub(r"n't", " not", x)
  x = re.sub(r"n'", "ng", x)
  x = re.sub(r"'bout", "about", x)
  x = re.sub(r"'til", "until", x)
  x = ' '.join(i.replace("'",'') for i in x.split(' ') if i ).strip()
  return x

def replace_in_conversations(conversations,unk_dict):
    new_conversations = []
    for conversation in conversations:
        new_conversation = []
        for phrase in conversation:
            new_phrase = ' '.join(unk_dict.get(i,i) for i in phrase.split(' '))
            if new_phrase:
                new_conversation.append(new_phrase)
        if new_conversation:
            new_conversations.append(new_conversation)
    return new_conversations