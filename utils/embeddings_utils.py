import numpy as np
import tqdm


def create_embeddings(token_embs, mapping, pad_token):
    emb_dim = len(list(token_embs.values())[0])
    new_embeddings = np.zeros(shape=(len(mapping), emb_dim))
    for k, v in mapping.items():
        if k == pad_token:
            new_embeddings[v] = np.zeros(shape=(emb_dim))
        else:
            vector = np.random.normal(size=(emb_dim))
            new_embeddings[v] = token_embs.get(k, vector)
    return new_embeddings


def get_emb_rep_glove(tokens, embeddings):
    dict_tokens = {}
    missing = []
    for w in tqdm.tqdm(tokens):
        if w in embeddings.keys():
            dict_tokens.update({w: embeddings[w]})
        elif w.lower() in embeddings.keys():
            dict_tokens.update({w: embeddings[w.lower()]})
        elif w.capitalize() in embeddings.keys():
            dict_tokens.update({w: embeddings[w.capitalize()]})
        else:
            missing.append(w)

    print('{} words where absent in embedding'.format(len(missing)))
    return dict_tokens, missing


def get_emb_rep(tokens, embeddings):
    dict_tokens = {}
    missing = []
    for w in tqdm.tqdm(tokens):
        try:
            try:
                dict_tokens.update({w: embeddings.word_vec(w.lower())})
            except:
                dict_tokens.update({w: embeddings.word_vec(w.captialize())})
        except:
            missing.append(w)

    print('{} words where absent in embedding'.format(len(missing)))
    return dict_tokens, missing
