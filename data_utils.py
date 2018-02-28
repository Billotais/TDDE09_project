import numpy as np
from gensim.models import KeyedVectors
import pickle
import json

def get_trees(data):
    for sentence in data:
        out = []
        out.append(0)
        for word in sentence:
            out.append(int(word[6]))
        yield out

def load_data(filename):
    data = [[]]
    with open(filename, encoding='utf-8') as source:
        for line in source:
            if len(line) == 1: 
                data.append([])
            elif line.split()[0].isdigit():
                data[-1].append(line.split('\t'))
    if not data[-1]:
        data = data[:-1]

    return data

def accuracy(preds, golds, correction=0):
    count = -correction
    score = -correction
    for pred, gold in zip(preds, golds):
        count += 1
        if pred == gold:
            score +=1
    return score/count

def get_sentences(data):
    sents = []
    max_len = 0
    for sent in data:
        out = []
        for word in sent:
            out.append(word[1])
        sents.append(out)
        max_len = max(max_len, len(out))
    return sents, max_len

def get_tags(data):
    tags = []
    for sent in data:
        out = []
        for word in sent:
            out.append(word[3])
        tags.append(out)
    return tags

def load_embeddings(voc_inv, params):
    embeddings = []
    model = KeyedVectors.load_word2vec_format(params['wv_location'], binary=True, limit=params['emb_limit'])
    for i, _ in enumerate(voc_inv):
        if voc_inv[i] in model:
            embeddings.append(model[voc_inv[i]])
        else:
            embeddings.append(np.random.uniform(-0.25, 0.25, 300))

    return embeddings

def get_tag_matrix(tag_dict):
    dim = len(tag_dict)
    tag_matrix = np.zeros((dim,dim))
    np.fill_diagonal(tag_matrix, 1)
    return tag_matrix

def build_dict(data):
    word_dict = {}
    word_dict_reverse = {}
    ind = 0
    for sent in data:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = ind
                word_dict_reverse[ind] = word
                ind+=1
    return word_dict, word_dict_reverse

def pad_sents(sents, seq_len, pad = '<PAD/>'):
    padded_sents = []
    sent_lens = []
    for i in range(len(sents)):
        sent = sents[i]
        sent_len = len(sent)
        num_pad = seq_len - sent_len
        padded_seq = sent + [pad] * num_pad
        padded_sents.append(padded_seq)
        sent_lens.append(sent_len)
    return padded_sents, sent_lens

def process_data(data, model_path):
    x, max_len_sents = get_sentences(data)
    x, sent_lens = pad_sents(x, max_len_sents)
    voc, voc_inv = build_dict(x)
    tags = get_tags(data)
    tags, _ = pad_sents(tags, max_len_sents)
    tag_dict, tag_dict_inv = build_dict(tags)
    x = np.array([[voc[word] for word in sentence] for sentence in x])
    tags = np.array([[tag_dict[word] for word in sentence] for sentence in tags])

    x_emb = load_embeddings(voc_inv,model_path)
    y = get_tag_matrix(tag_dict_inv)

    return x_emb, x, y, tags, voc, voc_inv, tag_dict_inv, sent_lens


def load_processed_data(trained_dir):
    voc = json.loads(open(trained_dir + 'voc.json',encoding='utf-8').read())
    voc_inv = json.loads(open(trained_dir + 'voc_inv.json',encoding='utf-8').read())
    tag_dict_inv = json.loads(open(trained_dir + 'tag_dict_inv.json',encoding='utf-8').read())

    with open(trained_dir + 'sent_lens.pickle', 'rb') as input_file:
        fetched_sent_lens = pickle.load(input_file)
    sent_lens = np.array(fetched_sent_lens, dtype = np.float32)

    with open(trained_dir + 'x_embeddings.pickle', 'rb') as input_file:
        fetched_x_embeddings = pickle.load(input_file)
    x_embeddings = np.array(fetched_x_embeddings, dtype = np.float32)

    with open(trained_dir + 'x.pickle', 'rb') as input_file:
        fetched_x = pickle.load(input_file)
    x = np.array(fetched_x, dtype = np.float32)

    with open(trained_dir + 'y_onehot.pickle', 'rb') as input_file:
        fetched_y_onehot = pickle.load(input_file)
    y_onehot = np.array(fetched_y_onehot, dtype = np.float32)

    with open(trained_dir + 'tags.pickle', 'rb') as input_file:
        fetched_tags = pickle.load(input_file)
    tags = np.array(fetched_tags, dtype = np.float32)

    return x_embeddings, x, y_onehot, tags, voc, voc_inv, tag_dict_inv, sent_lens


